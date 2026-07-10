#!/usr/bin/env python3
"""Offline EAGLE3 acceptance eval v2 — WITH prompt-KV prefill.

The v1 harness (eagle3_offline_accept.py) replayed each round from an EMPTY
drafter cache, so the drafter was starved of the context it was trained to
condition on (=> 0-0.15 accept on everything). It also compared draft[i]
against target[i+1] (off-by-one vs the live _eagle3_walk, which aligns
draft[i] <-> target[i]).

v2 reproduces the LIVE drafter state machine faithfully, per request:

  1. PREFILL the drafter KV from the request's prompt_*.npz (concatenated
     prompt-phase target hiddens + prompt ids + first token), exactly like
     m3_eagle3.py's `drafter.prefill_from_target_hidden(...)`.
  2. Replay rounds IN ORDER, maintaining drafter KV across rounds:
       - draft_block() with the config under test  -> candidate tokens
       - score candidate[i] == target_tokens[i] consecutively from 0
         (the live _eagle3_walk metric)
       - advance the drafter along the TRUE trajectory using the RECORDED
         (verify_hidden, live_accepted, live new_tokens) via
         accept_verified_tokens(), so every round's recorded inputs stay
         valid for the next round.

Scoring caveat (documented in the results file): the dumps were captured on
a ~1.23 accept/round trajectory, so target_tokens[i] for i > live_accepted
are conditioned on REJECTED live drafts and are not the true continuation.
We therefore report:
  - raw     : consecutive draft==target matches (exact live-loop metric)
  - trusted : min(raw, live_accepted+1)  (only credit up to the correction
              token, the last reliable position)
  - p0      : fraction of rounds with draft[0]==target[0] — fully reliable
              (target[0] depends only on the bonus token, never on drafts).

Swept axes (all cheap, one bf16 drafter reused; norm_before_residual is a
runtime flag on layers[0], seg/fc-norm perms are attributes read by an
overridden _prepare_target_hidden):
  - prompt-KV prefill on / off        (proves the prefill matters)
  - norm_before_residual off / on
  - seg_perm : which of the 3 dumped 6144-chunks feeds fc-block k
               (fc_norm[k] follows the chunk) -> tests capture concat order
               / consistent capture<->(fc_norm,fc-block) relabeling
  - fc_norm-only perm (seg fixed)     -> tests fc_norm mislabel vs fc block

Usage:
  mlx-python ops/eagle3_offline_accept2.py [dump_dir] [--quick]
"""
import glob
import itertools
import json
import os
import sys

import numpy as np
import mlx.core as mx
import mlx.nn as nn

_DRAFT_PATH = os.environ.get(
    "MLX_M3_EAGLE3_DRAFT",
    os.path.expanduser("~/.exo/models/Inferact--MiniMax-M3-EAGLE3"),
)
_TH = 6144  # target_hidden_size per capture (3 captures -> 18432)


# --------------------------------------------------------------------------
# Drafter: TorchSpec adapter (mirrors m3_eagle3.py) with SWAPPABLE seg/fc-norm
# permutations so a single loaded instance covers the whole sweep.
# --------------------------------------------------------------------------
def build_drafter():
    from mlx_vlm.speculative.drafters.eagle3.eagle3 import Eagle3DraftModel
    from mlx_vlm.speculative.drafters.eagle3.config import Eagle3Config, TextConfig

    class TorchSpecEagle3(Eagle3DraftModel):
        def __init__(self, config):
            super().__init__(config)
            text = config.transformer_layer_config
            self.fc_norm = [
                nn.RMSNorm(self.target_hidden_size, eps=text.rms_norm_eps)
                for _ in range(3)
            ]
            # runtime-swappable config (read by _prepare_target_hidden):
            #   part_k = fc_norm[fcnorm_perm[k]]( chunk[ seg_perm[k] ] )
            #   feed  = fc( concat(parts) )
            self._seg_perm = (0, 1, 2)
            self._fcnorm_perm = (0, 1, 2)

        def _prepare_target_hidden(self, hidden):
            if hidden.shape[-1] == self.hidden_size:
                return hidden
            th = self.target_hidden_size
            parts = []
            for k in range(3):
                s = self._seg_perm[k]
                chunk = hidden[..., s * th : (s + 1) * th]
                parts.append(self.fc_norm[self._fcnorm_perm[k]](chunk))
            return self.fc(mx.concatenate(parts, axis=-1))

        def bind(self, target_model):  # rank-0 keeps its own embedding
            return self

    with open(os.path.join(_DRAFT_PATH, "config.json")) as f:
        hf = json.load(f)
    text = TextConfig(
        model_type="llama",
        hidden_size=int(hf["hidden_size"]),
        intermediate_size=int(hf["intermediate_size"]),
        num_hidden_layers=int(hf.get("num_hidden_layers", 1)),
        num_attention_heads=int(hf["num_attention_heads"]),
        num_key_value_heads=int(
            hf.get("num_key_value_heads", hf["num_attention_heads"])
        ),
        head_dim=int(hf.get("head_dim") or 0) or None,
        rms_norm_eps=float(hf.get("rms_norm_eps", 1e-6)),
        vocab_size=int(hf["vocab_size"]),
        max_position_embeddings=int(hf.get("max_position_embeddings", 1048576)),
        rope_theta=float(hf.get("rope_theta", 5000000)),
        attention_bias=bool(hf.get("attention_bias", False)),
        hidden_act=str(hf.get("hidden_act", "silu")),
        tie_word_embeddings=bool(hf.get("tie_word_embeddings", False)),
    )
    cfg = Eagle3Config(
        model_type="eagle3",
        transformer_layer_config=text,
        draft_vocab_size=int(hf.get("draft_vocab_size", hf["vocab_size"])),
        target_hidden_size=int(hf["hidden_size"]),
        tie_word_embeddings=bool(hf.get("tie_word_embeddings", False)),
        norm_before_residual=False,  # toggled at runtime on layers[0]
        norm_before_fc=False,        # TorchSpec per-capture norms in subclass
        eagle_aux_hidden_state_layer_ids=[2, 30, 57],
        block_size=4,
    )
    model = TorchSpecEagle3(cfg)
    weights = {}
    for fp in sorted(glob.glob(os.path.join(_DRAFT_PATH, "*.safetensors"))):
        weights.update(mx.load(fp))
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model


def reset_drafter(d):
    """Fresh per (request, config): empty KV, clear seeds/positions."""
    d._cache = d.make_cache()
    d._seed_token = None
    d._seed_hidden = None
    d._next_position = 1
    d._round_appended = 0
    d.accept_lens = []
    d.draft_lens = []


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_request(req_dir):
    pf = sorted(glob.glob(os.path.join(req_dir, "prompt_*.npz")))
    prompt = None
    if pf:
        zp = np.load(pf[0])
        prompt = {
            "hidden": mx.array(zp["prompt_hidden"]).astype(mx.bfloat16),  # (1,n,18432)
            "ids": mx.array(zp["prompt_ids"]).reshape(1, -1).astype(mx.int32),
            "first": int(zp["first_token"]),
        }
    rounds = []
    for f in sorted(glob.glob(os.path.join(req_dir, "round_*.npz"))):
        z = np.load(f)
        rounds.append({
            "hidden": mx.array(z["verify_hidden"]).astype(mx.bfloat16),  # (1,K+1,18432)
            "draft": [int(x) for x in np.asarray(z["draft_tokens"]).reshape(-1)],
            "target": [int(x) for x in np.asarray(z["target_tokens"]).reshape(-1)],
            "bonus": int(z["bonus_in"]),
            "live_acc": int(z["accepted"]),
        })
    return prompt, rounds


# --------------------------------------------------------------------------
# Replay + score one request under one config
# --------------------------------------------------------------------------
SAMPLER = lambda lg: mx.argmax(lg, axis=-1)


def consec_match(a, b):
    n = 0
    for x, y in zip(a, b):
        if x == y:
            n += 1
        else:
            break
    return n


def replay_request(d, prompt, rounds, *, prefill):
    """Return per-round list of dicts: raw, trusted, live_acc, repro (my draft
    row == recorded live draft row)."""
    reset_drafter(d)
    n_draft = 3  # block_size 4 -> 3 candidate tokens (matches dumps)

    if prefill and prompt is not None:
        d.prefill_from_target_hidden(
            prompt["ids"], prompt["hidden"], prompt["first"],
            SAMPLER, mx.int32, greedy=True,
        )
        b = prompt["first"]
        hidden = prompt["hidden"][:, -1:, :]
    else:
        b = rounds[0]["bonus"] if rounds else 0
        hidden = rounds[0]["hidden"][:, :1, :] if rounds else None

    out = []
    for idx, r in enumerate(rounds):
        # --- draft with the config under test ---
        block = d.draft_block(b, hidden, d._cache, 4, SAMPLER, mx.int32, greedy=True)
        my_row = [int(v) for v in block.reshape(-1).tolist()][:n_draft]
        raw = consec_match(my_row, r["target"][:n_draft])
        la = r["live_acc"]
        trusted = min(raw, la + 1)
        repro = 1 if my_row == r["draft"][:n_draft] else 0
        out.append({"raw": raw, "trusted": trusted, "live_acc": la,
                    "repro": repro, "my": my_row, "idx": idx})

        # --- advance along the TRUE (recorded) trajectory ---
        live_draft = mx.array([r["draft"][:n_draft]], dtype=mx.int32)
        live_new = r["draft"][:la] + [r["target"][la]]
        d.accept_verified_tokens(
            r["hidden"], live_draft, la, live_new, SAMPLER, mx.int32, greedy=True
        )
        b = live_new[-1]
        hidden = r["hidden"][:, la : la + 1, :]
    return out


# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------
def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    quick = "--quick" in sys.argv
    dump = argv[0] if argv else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "logs", "e3dump3"
    )
    req_dirs = sorted(
        os.path.join(dump, d) for d in os.listdir(dump)
        if d.startswith("req_") and os.path.isdir(os.path.join(dump, d))
    )
    if quick:
        req_dirs = req_dirs[:4]
    print(f"dump={dump}  requests={len(req_dirs)}")

    d = build_drafter()
    perms = list(itertools.permutations((0, 1, 2)))

    # Config grid. Each config: (label, prefill, norm_res, seg_perm, fcnorm_perm)
    configs = []
    for prefill in (True, False):
        for nr in (False, True):
            for sp in perms:
                configs.append((
                    f"prefill={int(prefill)} norm_res={int(nr)} "
                    f"seg={sp} fcnorm=(0,1,2)",
                    prefill, nr, sp, (0, 1, 2),
                ))
    # fc_norm-only permutation axis (seg fixed identity, prefill on, both nr)
    for nr in (False, True):
        for fp in perms:
            if fp == (0, 1, 2):
                continue
            configs.append((
                f"prefill=1 norm_res={int(nr)} seg=(0,1,2) fcnorm={fp}",
                True, nr, (0, 1, 2), fp,
            ))

    # Accumulators keyed by label
    agg = {c[0]: [] for c in configs}         # list of per-round dicts (all reqs)
    per_req = {c[0]: {} for c in configs}     # label -> req_name -> mean raw

    live_rounds_all = []  # recorded live accepted, for the baseline anchor

    for rd in req_dirs:
        name = os.path.basename(rd)
        prompt, rounds = load_request(rd)
        live_rounds_all += [r["live_acc"] for r in rounds]
        for (label, prefill, nr, sp, fp) in configs:
            d.layers[0].norm_before_residual = nr
            d._seg_perm = sp
            d._fcnorm_perm = fp
            res = replay_request(d, prompt, rounds, prefill=prefill)
            agg[label] += res
            per_req[label][name] = (
                sum(x["raw"] for x in res) / max(1, len(res))
            )
        del prompt, rounds
        mx.clear_cache()
        print(f"  done {name}")

    # ---- summarize ----
    def summarize(rows):
        n = max(1, len(rows))
        raw = sum(x["raw"] for x in rows) / n
        trusted = sum(x["trusted"] for x in rows) / n
        p0 = sum(1 for x in rows if x["raw"] >= 1) / n
        p1 = sum(1 for x in rows if x["raw"] >= 2) / n
        p2 = sum(1 for x in rows if x["raw"] >= 3) / n
        repro = sum(x["repro"] for x in rows) / n
        return raw, trusted, p0, p1, p2, repro

    ranked = sorted(
        ((label, *summarize(agg[label])) for label in agg),
        key=lambda t: t[1], reverse=True,
    )
    live_mean = sum(live_rounds_all) / max(1, len(live_rounds_all))

    print(f"\n=== live recorded accept/round = {live_mean:.3f} "
          f"({len(live_rounds_all)} rounds) ===\n")
    print(f"{'config':60s} {'raw':>6s} {'trust':>6s} {'p0':>5s} "
          f"{'p1':>5s} {'p2':>5s} {'repro':>6s}")
    for label, raw, tr, p0, p1, p2, repro in ranked:
        print(f"{label:60s} {raw:6.3f} {tr:6.3f} {p0:5.2f} "
              f"{p1:5.2f} {p2:5.2f} {repro:6.2f}")

    write_results(dump, ranked, per_req, agg, live_mean, len(live_rounds_all),
                  len(req_dirs), summarize)


def write_results(dump, ranked, per_req, agg, live_mean, n_rounds, n_req,
                  summarize):
    out = os.path.join(dump, "ACCEPTANCE_RESULTS.md")
    L = []
    W = L.append
    W("# EAGLE3 offline acceptance sweep — MiniMax-M3 (e3dump3)\n")
    W(f"- drafter: `{_DRAFT_PATH}`")
    W(f"- data: `{dump}`  ({n_req} requests, {n_rounds} rounds)")
    W(f"- harness: `ops/eagle3_offline_accept2.py` "
      f"(prompt-KV prefill + faithful cross-round drafter KV)")
    W(f"- **live recorded accept/round = {live_mean:.3f}** "
      f"(mean of dumped `accepted`; anchors the ~1.18 live number)\n")
    W("## Metrics")
    W("- `raw` — consecutive draft==target_argmax matches from position 0 "
      "(exact live `_eagle3_walk` metric).")
    W("- `trusted` — `min(raw, live_accepted+1)`. The dumps were captured on "
      "a low-acceptance trajectory, so `target_tokens[i]` for "
      "`i > live_accepted` are conditioned on REJECTED live drafts and are not "
      "the true continuation; only positions up to the correction token are "
      "reliable ground truth. Measurable mean ceiling here ~"
      f"{live_mean + 1:.2f}.")
    W("- `p0` — fraction of rounds with `draft[0]==target[0]`. **Fully "
      "reliable** (target[0] depends only on the bonus token, never on "
      "drafts). Best single discriminator.")
    W("- `p1`,`p2` — fraction reaching >=2, >=3 raw matches.")
    W("- `repro` — fraction of rounds where the replayed draft row exactly "
      "equals the RECORDED live draft row (harness-faithfulness check for the "
      "live-equivalent config).\n")
    W("## Ranked configs (by trusted; raw shown too)\n")
    W("| rank | config | raw | trusted | p0 | p1 | p2 | repro |")
    W("|---:|---|---:|---:|---:|---:|---:|---:|")
    for i, (label, raw, tr, p0, p1, p2, repro) in enumerate(ranked, 1):
        W(f"| {i} | {label} | {raw:.3f} | {tr:.3f} | {p0:.2f} | "
          f"{p1:.2f} | {p2:.2f} | {repro:.2f} |")
    W("")

    # per-request breakdown for the top 3
    W("## Per-request raw accept/round — top 3 configs\n")
    top = [r[0] for r in ranked[:3]]
    reqs = sorted(next(iter(per_req.values())).keys())
    W("| request | " + " | ".join(f"#{i+1}" for i in range(len(top))) + " |")
    W("|---|" + "---:|" * len(top))
    for rq in reqs:
        W(f"| {rq} | " + " | ".join(f"{per_req[t][rq]:.2f}" for t in top) + " |")
    W("")
    for i, t in enumerate(top, 1):
        W(f"- **#{i}** = `{t}`")
    W("")

    # with vs without prefill, matched otherwise
    W("## Prefill ablation (matched norm_res / seg / fc-norm)\n")
    W("| config (sans prefill flag) | prefill=1 raw | prefill=0 raw | delta |")
    W("|---|---:|---:|---:|")
    seen = set()
    for label in agg:
        if not label.startswith("prefill=1 "):
            continue
        base = label[len("prefill=1 "):]
        off = "prefill=0 " + base
        if off not in agg or base in seen:
            continue
        seen.add(base)
        on_raw = summarize(agg[label])[0]
        off_raw = summarize(agg[off])[0]
        W(f"| {base} | {on_raw:.3f} | {off_raw:.3f} | "
          f"{on_raw - off_raw:+.3f} |")
    W("")

    # decay diagnostic: raw accept vs round-index bucket for the top config
    W("## Round-position decay — top config "
      "(flat/uniform => distribution shift; decaying => KV/positional)\n")
    top_label = ranked[0][0]
    rows = agg[top_label]
    buckets = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 10**9)]
    W(f"top config: `{top_label}`\n")
    W("| round idx bucket | n | mean raw |")
    W("|---|---:|---:|")
    for lo, hi in buckets:
        sel = [x["raw"] for x in rows if lo <= x["idx"] < hi]
        if sel:
            W(f"| {lo}-{hi if hi < 10**9 else '+'} | {len(sel)} | "
              f"{sum(sel)/len(sel):.3f} |")
    W("")
    with open(out, "w") as f:
        f.write("\n".join(L))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
