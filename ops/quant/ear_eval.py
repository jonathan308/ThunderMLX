#!/usr/bin/env python3
"""Offline EAR (Expected Acceptance Rate) evaluator for quant arms.

Runs a teacher-forced forward pass of a quantized MiniMax-M3 checkpoint over a
fixed, deterministic token corpus WITHOUT holding the model resident: each
layer is materialized from disk (CPU-stream copy, the int32-safe pattern from
m3_mixed_quant.py), run on GPU, then freed. Captures the top-K next-token
distribution at every position; ear_compare.py computes pairwise EAR
(sum of min(p,q) over the union of top-10 sets — arXiv 2605.02404's metric).

The corpus is built deterministically from fixed local files, so every arm
sees byte-identical token streams. B equal-length sequences, no padding.

Usage:
  ear_eval.py --model ~/.exo/models/MiniMax-M3-mixed-4.5 \
              --out /Volumes/Models/ear_mixed45.npz
"""
import argparse
import json
import os

import numpy as np

os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "16")
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "512")

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map, tree_unflatten

SEQ_LEN = 384
TOP_K = 32
MAX_EVAL_ELEMS = 1_500_000_000  # int32-overflow ceiling (see m3_mixed_quant.py)

_CL = os.path.expanduser("~/minimax-m3-cluster")
CORPUS_FILES = [
    # (path, offsets) — fixed slices; DO NOT reorder or the streams change.
    (f"{_CL}/sharded_server.py",
     [0, 100_000, 200_000, 300_000, 400_000, 500_000, 600_000, 700_000]),
    (f"{_CL}/m3_pipeline_patch.py", [0, 12_000]),
    (f"{_CL}/README.md", [0]),
    (f"{_CL}/ops/quant/m3_mixed_quant.py", [0, 8_000]),
    (f"{_CL}/overthink/ab_bench.py", [0, 8_000]),
    (f"{_CL}/dashboard.html", [0, 20_000, 40_000]),
    (f"{_CL}/cluster_gui.py", [0, 15_000, 30_000]),
    (f"{_CL}/launch_cluster.sh", [0]),
    (f"{_CL}/overthink/markers.json", [0]),
]
CHAT_PROMPTS = [
    "Write a Python function that finds the longest palindromic substring, "
    "with O(n^2) time and O(1) extra space. Include tests.",
    "Explain the difference between pipeline and tensor parallelism for "
    "serving a large mixture-of-experts model on two machines.",
    "Debug this: a distributed MLX cluster wedges when rank1 holds a stale "
    "RDMA queue pair after a crashed launch. Outline a recovery runbook.",
    "Summarize the tradeoffs of 4-bit group-32 vs 4-bit group-64 affine "
    "quantization for a 429B MoE language model.",
]


def build_corpus(tokenizer):
    seqs = []
    for path, offsets in CORPUS_FILES:
        raw = open(path, encoding="utf-8", errors="ignore").read()
        for off in offsets:
            chunk = raw[off:off + 6000]
            ids = tokenizer.encode(chunk)
            if len(ids) >= SEQ_LEN:
                seqs.append(ids[:SEQ_LEN])
    for p in CHAT_PROMPTS:
        msgs = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False)
        ids = tokenizer.encode(text)
        pad = tokenizer.encode(
            "Thinking about the request step by step, considering the "
            "constraints carefully and planning the answer structure. " * 40)
        ids = (list(ids) + pad)[:SEQ_LEN]
        seqs.append(ids)
    arr = np.array(seqs, dtype=np.int32)
    print(f"corpus: {arr.shape[0]} sequences x {SEQ_LEN} tokens")
    return arr


def build_ref_switch_mlp(layer_idx, m, kind, src_index, src_dir, load_cache):
    """8b/g64-quantized fused expert module built per-expert straight from the
    bf16 source files — the int32-safe route (mirrors m3_mixed_quant's
    emit_switch_mlp). Everything else in the reference stays native bf16, so
    the reference is 'bf16 with distribution-lossless 8-bit experts'."""
    from mlx_vlm.models.switch_layers import QuantizedSwitchLinear

    def load_src(name):
        sh = src_index[name]
        if sh not in load_cache:
            load_cache.clear()
            load_cache[sh] = mx.load(os.path.join(src_dir, sh))
        return load_cache[sh][name]

    pre = f"language_model.model.layers.{layer_idx}.block_sparse_moe"
    n_total = m.weight.shape[0]
    parts = []
    for e in range(n_total):
        if kind == "gate_up_proj":
            if e < n_total - 1:
                srcw = mx.concatenate(
                    [load_src(f"{pre}.experts.{e}.w1.weight"),
                     load_src(f"{pre}.experts.{e}.w3.weight")], axis=0)
            else:
                srcw = mx.concatenate(
                    [load_src(f"{pre}.shared_experts.gate_proj.weight"),
                     load_src(f"{pre}.shared_experts.up_proj.weight")], axis=0)
        else:
            srcw = (load_src(f"{pre}.experts.{e}.w2.weight") if e < n_total - 1
                    else load_src(f"{pre}.shared_experts.down_proj.weight"))
        srcw = mx.add(srcw, mx.array(0, dtype=srcw.dtype), stream=mx.cpu)
        mx.eval(srcw)
        p = mx.quantize(srcw, 64, 8)
        mx.eval(*p)
        parts.append(p)
        del srcw
    out = tuple(mx.stack([p[i] for p in parts], axis=0) for i in range(3))
    mx.eval(*out)
    num_experts, output_dims, input_dims = m.weight.shape
    q = QuantizedSwitchLinear(input_dims, output_dims, num_experts, False, 64, 8)
    q.weight, q.scales, q.biases = out
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bf16-ref", action="store_true",
                    help="model dir is the bf16 master: keep everything native "
                         "except experts on-the-fly 8b/g64 (near-lossless ref)")
    args = ap.parse_args()

    mx.set_default_device(mx.cpu)
    mx.set_cache_limit(8 * 1024 * 1024 * 1024)

    from mlx_vlm.utils import get_model_path, load_model, load_processor
    model_path = get_model_path(args.model)
    model = load_model(model_path, lazy=True, strict=False)
    processor = load_processor(model_path, True)
    tokenizer = getattr(processor, "tokenizer", processor)

    ids_np = build_corpus(tokenizer)
    ids = mx.array(ids_np)

    lm = model.language_model
    inner = lm.model

    def materialize_module(m):
        params = tree_flatten(m.parameters())
        mats = []
        for sub, a in params:
            if a.size > MAX_EVAL_ELEMS:
                raise RuntimeError(f"{sub}: exceeds int32-safe eval ceiling")
            r = mx.add(a, mx.array(0, dtype=a.dtype), stream=mx.cpu)
            mats.append((sub, r))
        mx.eval([r for _, r in mats])
        m.update(tree_unflatten(mats))

    def free_module(m):
        m.update(tree_map(lambda _: mx.array([]), m.parameters()))
        mx.clear_cache()

    from mlx_vlm.models.minimax_m3_vl.language import create_attention_mask

    materialize_module(inner.embed_tokens)
    with mx.stream(mx.gpu):
        h = inner.embed_tokens(ids)
    mx.eval(h)
    free_module(inner.embed_tokens)

    with mx.stream(mx.gpu):
        mask = create_attention_mask(h, None)
        if mask is not None:
            mx.eval(mask)

    src_index, load_cache = None, {}
    if args.bf16_ref:
        src_index = json.load(open(os.path.join(
            str(model_path), "model.safetensors.index.json")))["weight_map"]

    import time
    t0 = time.time()
    for i, layer in enumerate(inner.layers):
        moe = getattr(layer, "block_sparse_moe", None)
        if args.bf16_ref and moe is not None and hasattr(moe, "switch_mlp"):
            sm = moe.switch_mlp
            for kind in ("gate_up_proj", "down_proj"):
                setattr(sm, kind, build_ref_switch_mlp(
                    i, getattr(sm, kind), kind, src_index, str(model_path),
                    load_cache))
        materialize_module(layer)
        with mx.stream(mx.gpu):
            h = layer(h, mask, None, position_ids=None)
        mx.eval(h)
        free_module(layer)
        if (i + 1) % 10 == 0:
            print(f"  layer {i+1}/{len(inner.layers)} ({time.time()-t0:.0f}s)",
                  flush=True)

    materialize_module(inner.norm)
    with mx.stream(mx.gpu):
        h = inner.norm(h)
    mx.eval(h)
    free_module(inner.norm)

    materialize_module(lm.lm_head)
    B, T, _ = h.shape
    top_ids = np.zeros((B, T, TOP_K), dtype=np.int32)
    top_probs = np.zeros((B, T, TOP_K), dtype=np.float32)
    CH = 64
    for t0c in range(0, T, CH):
        with mx.stream(mx.gpu):
            logits = lm.lm_head(h[:, t0c:t0c + CH]).astype(mx.float32)
            probs = mx.softmax(logits, axis=-1)
            part = mx.argpartition(-probs, TOP_K - 1, axis=-1)[..., :TOP_K]
            pp = mx.take_along_axis(probs, part, axis=-1)
        mx.eval(part, pp)
        top_ids[:, t0c:t0c + CH] = np.array(part, copy=False)
        top_probs[:, t0c:t0c + CH] = np.array(pp, copy=False)
    free_module(lm.lm_head)

    np.savez_compressed(args.out, ids=ids_np, top_ids=top_ids,
                        top_probs=top_probs,
                        meta=json.dumps({"model": args.model, "seq_len": SEQ_LEN,
                                         "top_k": TOP_K}))
    print(f"EAR EVAL DONE: wrote {args.out} "
          f"({B}x{T} positions, top-{TOP_K}) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
