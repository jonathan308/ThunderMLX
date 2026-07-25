#!/usr/bin/env python3
"""Mixed-precision MiniMax-M3 quantization — the anti-overthink precision mix.

Tier ladder (2026-07-23, informed by arXiv 2606.00206 + a week of production
incident analysis + the model's own design review):

  8-bit g64 : lm_head (final logits — the literal overthinking mechanism),
              router gates (discrete expert selection),
              MSA indexer projections (sparse KV-block selection)
  6-bit g64 : embeddings, all attention q/k/v/o, the 3 dense-MLP layers
  4-bit g32 : bulk experts (switch_mlp) — group 32 instead of 64 halves
              within-group rounding error for ~+1 GB/layer, the cheapest
              quality lever on the bulk
  untouched : vision tower, projectors, norms (matches the existing 4-bit)

Expected output ≈ 250-260 GB (~4.7 bpw effective) — fits the 38/22 pipeline
split under the 254/120 GB wired ceilings.

Usage (from a shell WITH /Volumes/Models access):
  dry run (no writes; decision table + size estimate from shard headers):
    m3_mixed_quant.py --dry-run
  full conversion (hours; run under nohup):
    m3_mixed_quant.py --run
"""
import argparse, json, os, struct, sys
from collections import defaultdict

# GPU-timeout fix (same class as the historical prefill crashes, see
# launch_cluster.sh "COMMAND BUFFER SIZING"): cap how much lazy quantization
# work lands in a single Metal command buffer, or the driver kills it at ~10s
# (kIOGPUCommandBufferCallbackErrorTimeout — reproduced on shard 1, 2026-07-24).
# Must be set BEFORE mlx import.
os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "8")
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "256")

SRC = os.environ.get("M3_BF16_DIR", "/Volumes/Models/MiniMax-M3-bf16")
DST = os.environ.get("M3_MIXED_OUT", "/Volumes/Models/MiniMax-M3-mixed-4.5")

GLOBAL_BITS, GLOBAL_GROUP = 4, 32  # applies to anything predicate passes True


def quant_predicate(path, module):
    """(module_path, module) -> False | True | {"bits": b, "group_size": g}.

    Paths here are module paths (no .weight suffix), e.g.
    language_model.model.layers.30.self_attn.q_proj
    """
    p = path
    # Vision stack stays unquantized (matches the production 4-bit).
    if p.startswith(("vision_tower", "multi_modal_projector", "patch_merge_mlp")):
        return False
    # Tier 1 — discrete-decision modules: 8-bit.
    if p.endswith("lm_head"):
        return {"bits": 8, "group_size": 64, "mode": "affine"}
    if p.endswith(".block_sparse_moe.gate"):
        return {"bits": 8, "group_size": 64, "mode": "affine"}
    if p.endswith((".self_attn.index_q_proj", ".self_attn.index_k_proj")):
        return {"bits": 8, "group_size": 64, "mode": "affine"}
    # Tier 2 — every-token modules: 6-bit.
    if p.endswith("embed_tokens"):
        return {"bits": 6, "group_size": 64, "mode": "affine"}
    if p.endswith((".self_attn.q_proj", ".self_attn.k_proj",
                   ".self_attn.v_proj", ".self_attn.o_proj")):
        return {"bits": 6, "group_size": 64, "mode": "affine"}
    if p.endswith((".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj")):
        return {"bits": 6, "group_size": 64, "mode": "affine"}
    # Tier 3 — bulk experts: 4-bit, group 32.
    if ".switch_mlp." in p:
        return {"bits": 4, "group_size": 32, "mode": "affine"}
    # Everything else quantizable: global 4/32.
    return True


# ---------------------------------------------------------------- dry run --
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4}


def read_shard_shapes(shard_path):
    with open(shard_path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return {k: v for k, v in hdr.items() if k != "__metadata__"}


def module_path_of(tensor_name):
    for suf in (".weight", ".bias"):
        if tensor_name.endswith(suf):
            return tensor_name[: -len(suf)]
    return tensor_name


def bpw_of(decision):
    """Effective bits/weight incl. affine scale+bias overhead (2xfp16/group)."""
    if decision is False:
        return 16.0
    bits, group = (GLOBAL_BITS, GLOBAL_GROUP) if decision is True else (
        decision["bits"], decision["group_size"])
    return bits + (32.0 / group)


def dry_run():
    idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
    shards = sorted(set(idx["weight_map"].values()))
    per_tier = defaultdict(lambda: [0, 0.0])   # label -> [tensors, out_GB]
    total_out = 0.0
    for shard in shards:
        for name, info in read_shard_shapes(os.path.join(SRC, shard)).items():
            shape, dtype = info["shape"], info["dtype"]
            params = 1
            for d in shape:
                params *= d
            mp = module_path_of(name)
            decision = quant_predicate(mp, None)
            if name.endswith(".bias") or len(shape) <= 1:
                decision = False  # biases/norms stay native
            bpw = bpw_of(decision)
            out_gb = params * bpw / 8 / 1e9
            if decision is False:
                label = "native"
            elif decision is True:
                label = f"{GLOBAL_BITS}b/g{GLOBAL_GROUP}(default)"
            else:
                label = f"{decision['bits']}b/g{decision['group_size']}"
            key = f"{label:16s} {classify(mp)}"
            per_tier[key][0] += 1
            per_tier[key][1] += out_gb
            total_out += out_gb
    print(f"{'tier / module class':60s} {'tensors':>8s} {'out GB':>9s}")
    for key in sorted(per_tier):
        t, gb = per_tier[key]
        print(f"{key:60s} {t:8d} {gb:9.2f}")
    print(f"\nTOTAL ESTIMATED OUTPUT: {total_out:.1f} GB "
          f"(target 250-260; ceilings 254+120 on 38/22 split)")
    return total_out


def classify(mp):
    for tag in ("lm_head", "embed_tokens", "block_sparse_moe.gate",
                "index_q_proj", "index_k_proj", "switch_mlp",
                "self_attn", ".mlp.", "vision", "projector", "patch_merge"):
        if tag in mp:
            return tag
    return "other"


# ------------------------------------------------------------------- run --
# mlx_vlm.convert() is unusable for this source, for two independent reasons
# found the hard way (2026-07-24):
#  1. Its whole-model lazy graph makes GPU quantize kernels read bf16 straight
#     out of mmapped USB pages (~105 MB/s) and the ~10s Metal watchdog kills
#     them (kIOGPUCommandBufferCallbackErrorTimeout, reproduced in isolation).
#  2. mx.quantize/mx.dequantize corrupt tensors above ~2^31 elements (int32
#     index overflow): at the real fused-expert shape (129,6144,6144)=4.87e9
#     elements, packed output is exact garbage from expert ~57 on (verified:
#     0.0 packed-word equality vs per-expert reference), on GPU and in
#     mx.dequantize on CPU. Chunking along axis 0 is mathematically exact
#     (quant groups run along the last axis only) and chunk-consistency was
#     verified at every probed offset.
# Hence this streaming converter: per module, materialize bf16 via a CPU-
# stream copy (page faults are legal there), quantize on CPU in <=1e9-element
# chunks (CPU quantize measured 1.4 GB/s vs the disk's 80 MB/s — never the
# bottleneck), write shards incrementally. No GPU in the pipeline at all.
# Output format byte-mirrors mlx_vlm quantize_model/save_weights.

# The int32 overflow poisons EVERY kernel evaluated above ~2^31 elements —
# quantize, and also the plain add/concatenate that materialization and the
# sanitize expert-stack rely on (observed: the two fused switch_mlp tensors,
# 4.87e9 and 2.44e9 elements, come out zero/garbage past ~2^30 bytes while
# every tensor <= 1.23e9 elements is bit-exact). Ceiling with safety margin:
MAX_EVAL_ELEMS = 1_500_000_000


PROD_4BIT_INDEX = os.environ.get(
    "M3_PROD_INDEX",
    os.path.expanduser("~/.exo/models/mlx-community--MiniMax-M3-4bit/"
                       "model.safetensors.index.json"))


def full_run(smoke_shards=0, plan_only=False):
    import copy as copy_mod
    import gc
    import glob as glob_mod
    import shutil
    import time
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_map, tree_unflatten
    from mlx_vlm.utils import (MAX_FILE_SIZE_GB, create_model_card,
                               fetch_from_hub, get_model_path, save_config)

    mx.set_default_device(mx.cpu)   # loads/sanitize/materialize all on CPU
    mx.set_cache_limit(8 * 1024 * 1024 * 1024)

    dst = DST + "-smoke" if smoke_shards else DST
    print(f"[stream-quant] {SRC} -> {dst} "
          f"(plan_only={plan_only} smoke_shards={smoke_shards})", flush=True)

    model_path = get_model_path(SRC)
    model, config, processor = fetch_from_hub(
        model_path, lazy=True, trust_remote_code=True)
    if getattr(model, "_is_text_model", False):
        raise RuntimeError("text-model wrapper path not handled by this tool")
    target = model

    cast_dtype = getattr(mx, config.get("torch_dtype") or "bfloat16")
    cast_pred = getattr(target, "cast_predicate", lambda _k: True)

    # -- predicate wrapping, mirroring mlx_vlm.quant_utils.quantize_model --
    quantized_config = copy_mod.deepcopy(config)
    quantized_config.setdefault("vision_config", {})
    quant_params = {"group_size": GLOBAL_GROUP, "bits": GLOBAL_BITS,
                    "mode": "affine"}
    quantized_config["quantization"] = dict(quant_params)

    def wrapped(path, m):
        if not hasattr(m, "to_quantized"):
            return False
        if m.weight.shape[-1] % GLOBAL_GROUP != 0:
            return False
        d = quant_predicate(path, m)
        if isinstance(d, dict):
            quantized_config["quantization"][path] = d
        return d

    # -- pass 1: walk leaf modules, record decisions + lazy output specs ----
    def walk_leaves(tree, prefix=""):
        import mlx.nn as nn
        if isinstance(tree, nn.Module):
            yield prefix, tree
        elif isinstance(tree, dict):
            for k, v in tree.items():
                yield from walk_leaves(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(tree, list):
            for i, v in enumerate(tree):
                yield from walk_leaves(v, f"{prefix}.{i}" if prefix else str(i))

    modules = list(walk_leaves(target.leaf_modules()))
    plan = []          # (path, module, decision, [(tensor_name, nbytes, dtype)])
    covered = set()

    def out_dtype(name, arr):
        if (mx.issubdtype(arr.dtype, mx.floating) and cast_pred(name)
                and arr.dtype != cast_dtype):
            return cast_dtype
        return arr.dtype

    for path, m in modules:
        params = tree_flatten(m.parameters())
        if not params:
            continue
        decision = wrapped(path, m)
        entries = []
        if decision is False:
            for sub, arr in params:
                name = f"{path}.{sub}"
                dt = out_dtype(name, arr)
                entries.append((name, arr.size * dt.size, str(dt)))
        else:
            kwargs = dict(decision) if isinstance(decision, dict) else dict(quant_params)
            qm = m.to_quantized(**kwargs)          # lazy — shapes only
            for sub, arr in tree_flatten(qm.parameters()):
                entries.append((f"{path}.{sub}", arr.nbytes, str(arr.dtype)))
            del qm
        for sub, _ in params:
            covered.add(f"{path}.{sub}")
        plan.append((path, m, decision, entries))

    # leftover params living on non-leaf modules (e_score_correction_bias)
    leftovers = [(name, arr) for name, arr in tree_flatten(target.parameters())
                 if name not in covered and arr.size > 0]
    lo_entries = [(name, arr.size * out_dtype(name, arr).size,
                   str(out_dtype(name, arr))) for name, arr in leftovers]
    if lo_entries:
        plan.append(("<leftovers>", None, False, lo_entries))

    # -- plan report + parity gate against the prod 4-bit name set ----------
    all_entries = [e for _, _, _, es in plan for e in es]
    plan_names = [n for n, _, _ in all_entries]
    total_out = sum(nb for _, nb, _ in all_entries)
    tiers = defaultdict(lambda: [0, 0.0])
    for path, _, decision, es in plan:
        gb = sum(nb for _, nb, _ in es) / 1e9
        if decision is False:
            label = "native"
        elif decision is True:
            label = f"{GLOBAL_BITS}b/g{GLOBAL_GROUP}(default)"
        else:
            label = f"{decision['bits']}b/g{decision['group_size']}"
        key = f"{label:20s} {classify(path)}"
        tiers[key][0] += 1
        tiers[key][1] += gb
    print(f"{'tier / module class':44s} {'modules':>8s} {'out GB':>9s}", flush=True)
    for key in sorted(tiers):
        c, gb = tiers[key]
        print(f"{key:44s} {c:8d} {gb:9.2f}", flush=True)
    print(f"PLAN TOTAL: {len(plan_names)} tensors, {total_out/1e9:.1f} GB", flush=True)

    if len(set(plan_names)) != len(plan_names):
        raise RuntimeError("duplicate tensor names in plan")
    if os.path.exists(PROD_4BIT_INDEX):
        prod_names = set(json.load(open(PROD_4BIT_INDEX))["weight_map"])
        missing, extra = prod_names - set(plan_names), set(plan_names) - prod_names
        if missing or extra:
            for n in sorted(missing)[:8]:
                print(f"  PARITY missing (prod has, plan lacks): {n}", flush=True)
            for n in sorted(extra)[:8]:
                print(f"  PARITY extra (plan has, prod lacks):  {n}", flush=True)
            raise RuntimeError(
                f"name-set parity vs prod failed: {len(missing)} missing / "
                f"{len(extra)} extra")
        print(f"PARITY OK: plan name set == prod 4-bit ({len(prod_names)} tensors)",
              flush=True)
    else:
        print(f"WARNING: prod index not found at {PROD_4BIT_INDEX}; "
              "skipping parity gate", flush=True)
    if plan_only:
        return

    # -- shard packing (greedy in plan order, mirrors make_shards) ----------
    max_bytes = MAX_FILE_SIZE_GB << 30
    shard_of, cur, cur_bytes = {}, 0, 0
    for name, nb, _ in all_entries:
        if cur_bytes + nb > max_bytes and cur_bytes > 0:
            cur, cur_bytes = cur + 1, 0
        shard_of[name] = cur
        cur_bytes += nb
    n_shards = cur + 1
    if smoke_shards:
        n_shards_run = min(smoke_shards, n_shards)
        print(f"SMOKE: writing first {n_shards_run} of {n_shards} shards", flush=True)
    else:
        n_shards_run = n_shards
    shard_fmt = "model-{:05d}-of-{:05d}.safetensors"
    pending = defaultdict(dict)                    # shard_idx -> {name: array}
    remaining = defaultdict(int)
    for name in plan_names:
        remaining[shard_of[name]] += 1
    weight_map, written = {}, [0]

    os.makedirs(dst, exist_ok=True)
    # Clean leavings of any crashed prior attempt (0-byte shards, AppleDouble
    # junk, stale index) — os.listdir, not glob, so ._ dotfiles are caught.
    for fn in os.listdir(dst):
        if (fn.endswith(".safetensors") or fn.startswith("._")
                or fn == "model.safetensors.index.json"):
            os.remove(os.path.join(dst, fn))
    t_start = time.time()
    src_read = [0.0]

    def materialize(name, arr):
        if arr.size > MAX_EVAL_ELEMS:
            raise RuntimeError(
                f"{name}: {arr.size} elements exceeds the int32-safe eval "
                "ceiling — must be handled by a chunked special case")
        if (mx.issubdtype(arr.dtype, mx.floating) and cast_pred(name)
                and arr.dtype != cast_dtype):
            r = arr.astype(cast_dtype, stream=mx.cpu)
        else:
            r = mx.add(arr, mx.array(0, dtype=arr.dtype), stream=mx.cpu)
        mx.eval(r)
        return r

    src_index = json.load(
        open(os.path.join(SRC, "model.safetensors.index.json")))["weight_map"]
    src_shard_cache = {}

    def load_src_tensor(name):
        sh = src_index[name]
        if sh not in src_shard_cache:
            src_shard_cache.clear()
            src_shard_cache[sh] = mx.load(os.path.join(SRC, sh))
        return src_shard_cache[sh][name]

    def emit_switch_mlp(path, m, kwargs):
        """Fused expert tensors exceed the int32-safe ceiling, so never touch
        the sanitize concat graph: rebuild per-expert from the bf16 source
        files (mirroring _sanitize_moe_weights order: routed 0..n-1 in
        experts.N.{w1,w3,w2}, shared expert last) and stack packed outputs
        (always far below the ceiling)."""
        n_total = m.weight.shape[0]
        prefix = path.rsplit(".switch_mlp.", 1)[0]
        kind = path.rsplit(".", 1)[1]
        parts = []
        for e in range(n_total):
            if kind == "gate_up_proj":
                if e < n_total - 1:
                    srcw = mx.concatenate(
                        [load_src_tensor(f"{prefix}.experts.{e}.w1.weight"),
                         load_src_tensor(f"{prefix}.experts.{e}.w3.weight")],
                        axis=0)
                else:
                    srcw = mx.concatenate(
                        [load_src_tensor(f"{prefix}.shared_experts.gate_proj.weight"),
                         load_src_tensor(f"{prefix}.shared_experts.up_proj.weight")],
                        axis=0)
            else:
                if e < n_total - 1:
                    srcw = load_src_tensor(f"{prefix}.experts.{e}.w2.weight")
                else:
                    srcw = load_src_tensor(f"{prefix}.shared_experts.down_proj.weight")
            srcw = materialize(f"{path}.weight[{e}]", srcw)
            p = mx.quantize(srcw, kwargs["group_size"], kwargs["bits"])
            mx.eval(*p)
            parts.append(p)
            src_read[0] += srcw.nbytes
            del srcw
        out = tuple(mx.stack([p[i] for p in parts], axis=0) for i in range(3))
        mx.eval(*out)
        del parts
        for sub, a in zip(("weight", "scales", "biases"), out):
            emit(f"{path}.{sub}", a)

    def emit(name, arr):
        si = shard_of[name]
        if si >= n_shards_run:
            return
        pending[si][name] = arr
        remaining[si] -= 1
        if remaining[si] == 0:
            p = os.path.join(dst, shard_fmt.format(si + 1, n_shards))
            mx.save_safetensors(p, pending[si], metadata={"format": "mlx"})
            for k in pending[si]:
                weight_map[k] = os.path.basename(p)
            written[0] += 1
            del pending[si]
            gc.collect()
            mx.clear_cache()
            el = time.time() - t_start
            frac = src_read[0] / (854.2e9) if src_read[0] else 0
            eta = el / frac - el if frac > 0.001 else 0
            print(f"shard {si+1}/{n_shards} written | src {src_read[0]/1e9:6.1f} GB "
                  f"({src_read[0]/el/1e6:6.1f} MB/s) | ETA {eta/60:5.1f} min",
                  flush=True)

    for path, m, decision, entries in plan:
        if path == "<leftovers>":
            for name, arr in leftovers:
                emit(name, materialize(name, arr))
            continue
        # Plan order == pack order, so shard indices are monotonic across
        # modules: once a module lies entirely past the window, all later
        # ones do too. A boundary module still runs; emit() drops its
        # out-of-window tensors.
        if min(shard_of[n] for n, _, _ in entries) >= n_shards_run:
            break
        kwargs = dict(decision) if isinstance(decision, dict) else dict(quant_params)
        if decision is not False and kwargs.get("mode", "affine") != "affine":
            raise RuntimeError(f"non-affine mode unsupported: {path}")
        if decision is not False and ".switch_mlp." in path:
            if len(entries) != 3:
                raise RuntimeError(f"{path}: expected weight/scales/biases only")
            emit_switch_mlp(path, m, kwargs)
            m.update(tree_map(lambda _: mx.array([]), m.parameters()))
            continue
        params = tree_flatten(m.parameters())
        src_read[0] += sum(a.nbytes for _, a in params)
        mats = [(sub, materialize(f"{path}.{sub}", a)) for sub, a in params]
        if decision is False:
            for sub, r in mats:
                emit(f"{path}.{sub}", r)
        else:
            m.update(tree_unflatten(mats))
            qm = m.to_quantized(**kwargs)   # structure only; its lazy
            # quantize graph is replaced below, never evaluated
            qw, qs, qb = mx.quantize(m.weight, kwargs["group_size"], kwargs["bits"])
            mx.eval(qw, qs, qb)
            qm.weight, qm.scales, qm.biases = qw, qs, qb
            qparams = tree_flatten(qm.parameters())
            mx.eval([a for _, a in qparams])
            for sub, a in qparams:
                emit(f"{path}.{sub}", a)
            del qm
        m.update(tree_map(lambda _: mx.array([]), m.parameters()))
        del mats

    if smoke_shards:
        print(f"SMOKE COMPLETE: {written[0]} shards in {dst}", flush=True)
        return

    # -- index / config / aux files (mirrors convert() + save_weights) ------
    index = {"metadata": {"total_size": total_out},
             "weight_map": {k: weight_map[k] for k in sorted(weight_map)}}
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=4)
    for pattern in ["*.py", "*.json"]:
        for fpath in glob_mod.glob(str(model_path / pattern)):
            if os.path.basename(fpath) == "model.safetensors.index.json":
                continue
            shutil.copy(fpath, dst)
    for item in model_path.iterdir():
        if item.is_dir():
            d = os.path.join(dst, item.name)
            if os.path.exists(d):
                shutil.rmtree(d)
            shutil.copytree(item, d)
    processor.save_pretrained(dst)
    quantized_config["quantization_config"] = quantized_config["quantization"]
    save_config(quantized_config, config_path=os.path.join(dst, "config.json"))
    create_model_card(dst, None)
    # ExFAT + macOS spawns an AppleDouble ._ twin for every file written;
    # they crash naive shard globs (mx.load can't parse them). Sweep last.
    for fn in os.listdir(dst):
        if fn.startswith("._"):
            os.remove(os.path.join(dst, fn))
    print("CONVERSION COMPLETE", flush=True)
    print(f"output weights: {total_out/1e9:.1f} GB across {n_shards} shards "
          f"| wall {(time.time()-t_start)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="size estimate from bf16 shard headers (no model load)")
    ap.add_argument("--plan", action="store_true",
                    help="build real module plan + parity gate vs prod, no writes")
    ap.add_argument("--smoke-shards", type=int, default=0,
                    help="stream-convert only the first N shards into DST-smoke")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        dry_run()
    elif args.plan:
        full_run(plan_only=True)
    elif args.smoke_shards:
        full_run(smoke_shards=args.smoke_shards)
    elif args.run:
        full_run()
    else:
        ap.print_help()
        sys.exit(1)
