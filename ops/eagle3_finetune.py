#!/usr/bin/env python3
"""Offline calibration fine-tune of the MiniMax-M3 EAGLE3 drafter.

Runs ENTIRELY OFFLINE on this Mac (no servers / HTTP / cluster). Reuses the
validated replay harness (eagle3_offline_accept2.py) for the drafter model,
loading, and acceptance eval so the tuned weights are judged with the exact
same metric as the hunt (mean raw accepted/round, config #1 = prefill on,
NORM_RESIDUAL=1, seg/fc-norm identity).

Training objective (standard EAGLE next-token CE, teacher-forced):
  At drafter position i the input is (token_i, target_hidden_i) where
  target_hidden_i is the concatenated 18432-d capture that PREDICTED token_i;
  the drafter must predict token_{i+1}. This is exactly what
  `prefill_from_target_hidden` feeds. We build, per request, ONE concatenated
  teacher-forced sequence:
      prompt-phase:  tokens = [ids[1:], first_token]         hiddens = prompt_hidden
      decode-phase:  per round append the TRUE-path new tokens
                     (target_tokens[0:accepted+1]) and their verify hiddens
                     (verify_hidden[0:accepted+1]).
  Because the drafter advances its KV by exactly (accepted+1) along the
  accepted trajectory each round, this single contiguous sequence with a
  causal mask is a FAITHFUL replay of the drafter's true-path state (not an
  approximation). Only the rejected speculative branches are dropped (they
  carry off-distribution / unreliable target labels anyway).

Memory discipline (hard budget < 40 GB; production holds ~160 GB separately):
  - frozen embed_tokens + lm_head stay bf16 (2.46 B params -> 4.9 GB);
  - trainable core (fc, fc_norm.{0,1,2}, layers.0.*, norm) is fp32 so AdamW at
    ~1e-5 LR does not underflow (bf16 master weights would round tiny updates
    away);
  - attention uses the flash path (mask="causal", no cache) => O(L) memory;
  - long prompts are tiled into <= L_MAX windows, capped per request so the two
    5.5k-token longctx requests do not swamp the corpus;
  - eval acceptance runs on a separate bf16 drafter (production dtype).

Usage:
  mlx-python ops/eagle3_finetune.py [--scope core|proj] [--epochs N] [--lr F]
"""
import argparse
import glob
import json
import os
import random
import sys
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map, tree_flatten

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import eagle3_offline_accept2 as a2  # validated model / loader / replay

DUMP = os.path.join(_HERE, "logs", "e3dump2")
OUT = os.path.join(DUMP, "finetune")
_TH = 6144  # per-capture hidden size (3 captures -> 18432)

# Held-out requests: one strong (tool-json ~2.23) + one weak (prose-dialog
# ~0.65). NEVER trained on; the decision metric is their acceptance.
HELDOUT = ["req_10_tool-json", "req_11_prose-dialog"]

L_MAX = 512      # max window length (bounds logits/attention memory)
N_CAP = 2        # max windows kept per request (last N; balances domains)


# --------------------------------------------------------------------------
# Dataset: build per-request teacher-forced sequences, then tile into windows.
# --------------------------------------------------------------------------
def build_request_sequence(req_dir):
    """Return (tokens[M-1], hidden[M-1,18432] fp32, labels[M-1]) numpy arrays,
    the faithful prompt+accepted-decode teacher-forced sequence for one
    request. Every returned position has a valid next-token label."""
    pf = sorted(glob.glob(os.path.join(req_dir, "prompt_*.npz")))
    if not pf:
        return None
    zp = np.load(pf[0])
    ids = np.asarray(zp["prompt_ids"]).reshape(-1).astype(np.int32)
    ph = np.asarray(zp["prompt_hidden"]).astype(np.float32)  # (1,n,18432)
    ph = ph[0]
    first = int(zp["first_token"])
    n = ph.shape[0]

    # prompt-phase pairs: token_i predicted by hidden_i, feed shifted ids.
    tok_p = np.concatenate([ids[1:], np.array([first], np.int32)])  # (n,)
    hid_p = ph                                                       # (n,18432)

    # decode-phase pairs along the TRUE accepted trajectory.
    tok_d, hid_d = [], []
    for f in sorted(glob.glob(os.path.join(req_dir, "round_*.npz"))):
        z = np.load(f)
        acc = int(z["accepted"])
        vh = np.asarray(z["verify_hidden"]).astype(np.float32)[0]    # (K+1,18432)
        tt = np.asarray(z["target_tokens"]).reshape(-1).astype(np.int32)
        k = acc + 1                          # true new tokens this round
        k = min(k, vh.shape[0], tt.shape[0])
        tok_d.append(tt[:k])
        hid_d.append(vh[:k])
    # capture-only decode pairs (feature/capture-only): native NON-speculative
    # decode, one file per request. hidden[t] is the capture that predicted
    # tokens[t+1], so feed the shifted stream as labels — same pairing as the
    # round_*.npz decode-phase, just without a verify/accept trajectory.
    for f in sorted(glob.glob(os.path.join(req_dir, "decode_*.npz"))):
        z = np.load(f)
        hd = np.asarray(z["hidden"]).astype(np.float32)             # (T,18432)
        tk = np.asarray(z["tokens"]).reshape(-1).astype(np.int32)   # (T+1,)
        k = min(hd.shape[0], tk.shape[0] - 1)
        if k <= 0:
            continue
        tok_d.append(tk[1:1 + k])
        hid_d.append(hd[:k])
    if tok_d:
        tok_d = np.concatenate(tok_d)
        hid_d = np.concatenate(hid_d, axis=0)
        tokens = np.concatenate([tok_p, tok_d])
        hidden = np.concatenate([hid_p, hid_d], axis=0)
    else:
        tokens, hidden = tok_p, hid_p

    # teacher forcing: input pos i -> label tokens[i+1]; drop unlabeled last.
    inp_tokens = tokens[:-1]
    inp_hidden = hidden[:-1]
    labels = tokens[1:]
    return inp_tokens, inp_hidden, labels


def tile_windows(seq, l_max=L_MAX, n_cap=N_CAP):
    """Tile (tokens, hidden, labels) into <= l_max windows, keep the LAST
    n_cap (most recent => includes the decode tail)."""
    tokens, hidden, labels = seq
    M = tokens.shape[0]
    wins = []
    for a in range(0, M, l_max):
        b = min(a + l_max, M)
        wins.append((tokens[a:b], hidden[a:b], labels[a:b]))
    if n_cap and len(wins) > n_cap:
        wins = wins[-n_cap:]
    return wins


def build_dataset(req_names):
    """List of windows (each a numpy tuple) across the given requests."""
    windows, stats = [], []
    for name in req_names:
        seq = build_request_sequence(os.path.join(DUMP, name))
        if seq is None:
            continue
        wins = tile_windows(seq)
        pos = sum(w[0].shape[0] for w in wins)
        windows.extend(wins)
        stats.append((name, len(wins), seq[0].shape[0], pos))
    return windows, stats


# --------------------------------------------------------------------------
# Model prep: fp32 trainable core, bf16 frozen embed+lm_head.
# --------------------------------------------------------------------------
def prepare_model(scope):
    """Build the TorchSpec drafter (bf16), set config #1 wiring, cast the core
    to fp32, freeze embed_tokens+lm_head (+ layer for scope='proj')."""
    model = a2.build_drafter()
    model.layers[0].norm_before_residual = True   # NORM_RESIDUAL=1
    model._seg_perm = (0, 1, 2)
    model._fcnorm_perm = (0, 1, 2)

    # Cast ONLY the trainable core to fp32 (AdamW master weights); leave the
    # 2.46 B embed_tokens + lm_head bf16 and untouched — casting them to fp32
    # even transiently would spike ~10 GB.
    for mod in (model.fc, model.norm, model.layers[0], *model.fc_norm):
        mod.update(tree_map(lambda p: p.astype(mx.float32), mod.parameters()))

    model.freeze()
    model.fc.unfreeze()
    for fnorm in model.fc_norm:
        fnorm.unfreeze()
    model.norm.unfreeze()
    if scope == "core":
        model.layers[0].unfreeze()
    mx.eval(model.parameters())
    return model


def forward_logits(model, tokens_i32, hidden_f32, pos_offset=1):
    """Standalone faithful replica of Eagle3FirstLayer + _logits under config
    #1 (identity seg/fc-norm, norm_before_residual). fp32 core with bf16
    embed/lm_head at the two boundaries. mask='causal' => flash attention."""
    th = model.target_hidden_size
    seg, fcn = model._seg_perm, model._fcnorm_perm
    parts = [
        model.fc_norm[fcn[k]](hidden_f32[..., seg[k] * th : (seg[k] + 1) * th])
        for k in range(3)
    ]
    feed = model.fc(mx.concatenate(parts, axis=-1))                 # (1,L,6144) fp32

    embeds = model.embed_tokens(tokens_i32).astype(mx.float32)      # bf16 -> fp32
    layer = model.layers[0]
    e = layer.input_layernorm(embeds)
    hn = layer.hidden_norm(feed)
    residual = hn if layer.norm_before_residual else feed
    h = mx.concatenate([e, hn], axis=-1)                            # (1,L,12288)
    h = layer.self_attn(h, mask="causal", cache=None, position_offset=pos_offset)
    h = residual + h
    residual = h
    h = layer.mlp(layer.post_attention_layernorm(h))
    h = residual + h                                                # (1,L,6144)

    h = model.norm(h)
    logits = model.lm_head(h.astype(model.lm_head.weight.dtype))    # -> bf16
    return logits.astype(mx.float32)


# --------------------------------------------------------------------------
# Acceptance eval on a separate bf16 drafter (production dtype).
# --------------------------------------------------------------------------
def core_bf16_dict(model):
    """Flatten the tuned CORE params (everything except embed/lm_head) to bf16
    for a partial load into the eval / checkpoint model."""
    flat = dict(tree_flatten(model.parameters()))
    return {
        k: v.astype(mx.bfloat16)
        for k, v in flat.items()
        if not (k.startswith("embed_tokens.") or k.startswith("lm_head."))
    }


def eval_acceptance(eval_model, heldout_reqs):
    """mean raw accepted/round per held-out request + aggregate, config #1."""
    eval_model.layers[0].norm_before_residual = True
    eval_model._seg_perm = (0, 1, 2)
    eval_model._fcnorm_perm = (0, 1, 2)
    per_req, all_rows = {}, []
    for name in heldout_reqs:
        prompt, rounds = a2.load_request(os.path.join(DUMP, name))
        res = a2.replay_request(eval_model, prompt, rounds, prefill=True)
        raws = [x["raw"] for x in res]
        per_req[name] = sum(raws) / max(1, len(raws))
        all_rows.extend(raws)
        del prompt, rounds
        mx.clear_cache()
    agg = sum(all_rows) / max(1, len(all_rows))
    return per_req, agg, len(all_rows)


# --------------------------------------------------------------------------
# CE over a set of windows (no grad) — for train/eval loss reporting.
# --------------------------------------------------------------------------
def dataset_ce(model, windows):
    tot, cnt = 0.0, 0
    for tks, hid, lab in windows:
        t = mx.array(tks[None], dtype=mx.int32)
        h = mx.array(hid[None], dtype=mx.float32)
        y = mx.array(lab[None], dtype=mx.int32)
        logits = forward_logits(model, t, h)
        ce = nn.losses.cross_entropy(logits, y, reduction="sum")
        mx.eval(ce)
        tot += float(ce)
        cnt += lab.shape[0]
        del t, h, y, logits, ce
    mx.clear_cache()
    return tot / max(1, cnt)


# --------------------------------------------------------------------------
# Train one scope
# --------------------------------------------------------------------------
def train(scope, epochs, lr, seed, log):
    log(f"\n{'='*72}\nTRAIN scope={scope} epochs={epochs} lr={lr:g} seed={seed}\n{'='*72}")
    random.seed(seed)
    mx.random.seed(seed)

    train_reqs = [
        os.path.basename(d)
        for d in sorted(glob.glob(os.path.join(DUMP, "req_*")))
        if os.path.isdir(d) and os.path.basename(d) not in HELDOUT
    ]
    train_wins, tstats = build_dataset(train_reqs)
    eval_wins, estats = build_dataset(HELDOUT)
    n_train_pos = sum(w[0].shape[0] for w in train_wins)
    n_eval_pos = sum(w[0].shape[0] for w in eval_wins)
    log(f"train requests={len(train_reqs)} windows={len(train_wins)} "
        f"positions={n_train_pos}")
    for nm, nw, seqlen, pos in tstats:
        log(f"  {nm:28s} seq={seqlen:5d} windows={nw} train_pos={pos}")
    log(f"held-out requests={HELDOUT} windows={len(eval_wins)} positions={n_eval_pos}")

    model = prepare_model(scope)
    mx.clear_cache()
    ntrain = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    log(f"trainable params = {ntrain/1e6:.1f} M  (scope={scope})")

    # eval drafter (bf16) built ONCE; reload core each epoch. Share the frozen
    # bf16 embed_tokens + lm_head with the train model (identical, never
    # updated) to save ~4.9 GB.
    eval_model = a2.build_drafter()
    eval_model.embed_tokens = model.embed_tokens
    eval_model.lm_head = model.lm_head
    mx.clear_cache()

    total_steps = max(1, len(train_wins) * epochs)
    warmup = max(1, total_steps // 20)
    sched = optim.join_schedules(
        [optim.linear_schedule(0.0, lr, warmup),
         optim.cosine_decay(lr, total_steps - warmup, end=lr * 0.1)],
        [warmup],
    )
    opt = optim.AdamW(learning_rate=sched, betas=[0.9, 0.95], weight_decay=0.0)

    def loss_fn(t, h, y):
        logits = forward_logits(model, t, h)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    lvg = nn.value_and_grad(model, loss_fn)

    # ---- baseline (epoch 0) acceptance + CE, tuned weights == original core.
    eval_model.load_weights(list(core_bf16_dict(model).items()), strict=False)
    per0, agg0, nro = eval_acceptance(eval_model, HELDOUT)
    tr_ce0 = dataset_ce(model, train_wins)
    ev_ce0 = dataset_ce(model, eval_wins)
    history = [("0(base)", tr_ce0, ev_ce0, dict(per0), agg0)]
    log(f"[epoch 0 baseline] train_ce={tr_ce0:.4f} eval_ce={ev_ce0:.4f}  "
        f"heldout_acc={ {k: round(v,3) for k,v in per0.items()} } agg={agg0:.3f} "
        f"({nro} rounds)")

    for ep in range(1, epochs + 1):
        order = list(range(len(train_wins)))
        random.shuffle(order)
        run_ce, run_n, t0 = 0.0, 0, time.time()
        for si, wi in enumerate(order):
            tks, hid, lab = train_wins[wi]
            t = mx.array(tks[None], dtype=mx.int32)
            h = mx.array(hid[None], dtype=mx.float32)
            y = mx.array(lab[None], dtype=mx.int32)
            loss, grads = lvg(t, h, y)
            grads, gnorm = optim.clip_grad_norm(grads, 1.0)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state, loss)
            run_ce += float(loss) * lab.shape[0]
            run_n += lab.shape[0]
            del t, h, y, loss, grads
            if (si + 1) % 8 == 0:
                mx.clear_cache()
        mx.clear_cache()
        train_ce = run_ce / max(1, run_n)

        # ---- eval this epoch ----
        eval_model.load_weights(list(core_bf16_dict(model).items()), strict=False)
        per, agg, _ = eval_acceptance(eval_model, HELDOUT)
        ev_ce = dataset_ce(model, eval_wins)
        peak = mx.get_peak_memory() / 1e9
        history.append((str(ep), train_ce, ev_ce, dict(per), agg))
        log(f"[epoch {ep}] train_ce={train_ce:.4f} eval_ce={ev_ce:.4f}  "
            f"heldout_acc={ {k: round(v,3) for k,v in per.items()} } agg={agg:.3f}  "
            f"lr={float(sched(opt.step)):.2e} gnorm={float(gnorm):.2f} "
            f"peak={peak:.1f}GB dt={time.time()-t0:.0f}s")

        save_checkpoint(model, scope, ep, log)

    del eval_model
    mx.clear_cache()
    return history, tstats, estats


def save_checkpoint(model, scope, ep, log):
    ckpt_dir = os.path.join(OUT, f"ckpt_{scope}_epoch{ep}")
    os.makedirs(ckpt_dir, exist_ok=True)
    flat = dict(tree_flatten(model.parameters()))
    weights = {k: v.astype(mx.bfloat16) for k, v in flat.items()}
    mx.eval(list(weights.values()))
    path = os.path.join(ckpt_dir, "model.safetensors")
    mx.save_safetensors(path, weights)
    # copy config so m3_eagle3.py loads it unchanged.
    src_cfg = os.path.join(a2._DRAFT_PATH, "config.json")
    with open(src_cfg) as f:
        cfg = f.read()
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        f.write(cfg)
    log(f"  saved {path}  ({len(weights)} tensors)")
    return ckpt_dir


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def write_report(runs, tstats, estats, args, log):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "TRAINING_REPORT.md")
    L = []
    W = L.append
    W("# EAGLE3 drafter calibration fine-tune — MiniMax-M3 (e3dump2)\n")
    W(f"- drafter: `{a2._DRAFT_PATH}`")
    W(f"- data: `{DUMP}` (13 requests, 475 rounds)")
    W(f"- harness: reuses `ops/eagle3_offline_accept2.py` (config #1: prefill on, "
      "NORM_RESIDUAL=1, seg/fc-norm identity)")
    W(f"- trained on {13 - len(HELDOUT)} requests; **held out (never trained): "
      f"{', '.join(HELDOUT)}**")
    W(f"- decision metric: mean **raw** accepted/round on the held-out requests, "
      "same metric as the hunt (offline baseline 1.535 aggregate).\n")

    W("## Dataset\n")
    W("Per-request teacher-forced sequence = prompt-phase pairs "
      "`(shifted ids, prompt_hidden)` ++ decode-phase TRUE-path pairs "
      "`(target_tokens[0:acc+1], verify_hidden[0:acc+1])`; tiled into "
      f"<= {L_MAX}-token windows, last {N_CAP} kept per request.\n")
    W("| request | seq len | windows | train positions |")
    W("|---|---:|---:|---:|")
    for nm, nw, seqlen, pos in tstats:
        W(f"| {nm} | {seqlen} | {nw} | {pos} |")
    tot_pos = sum(s[3] for s in tstats)
    tot_win = sum(s[1] for s in tstats)
    W(f"| **train total** | | **{tot_win}** | **{tot_pos}** |")
    for nm, nw, seqlen, pos in estats:
        W(f"| {nm} (held-out) | {seqlen} | {nw} | {pos} |")
    W("")

    W("## Hyperparameters\n")
    W(f"- optimizer: AdamW(betas=(0.9,0.95), wd=0.0), grad-clip 1.0")
    W(f"- lr: {args.lr:g} (linear warmup ~5% then cosine decay to 10%)")
    W(f"- epochs: {args.epochs}; batch: 1 window/step")
    W(f"- dtype: trainable core fp32 (AdamW master), frozen embed+lm_head bf16; "
      "eval/checkpoint bf16")
    W(f"- seed: {args.seed}\n")

    for scope, (history, _, _) in runs.items():
        W(f"## Run: scope=`{scope}`\n")
        if scope == "core":
            W("trains `fc, fc_norm.{0,1,2}, layers.0.*, norm` "
              "(embed+lm_head frozen).\n")
        else:
            W("trains `fc, fc_norm.{0,1,2}, norm` ONLY (transformer layer + "
              "embed + lm_head frozen — minimal calibration, lowest overfit "
              "capacity).\n")
        W("| epoch | train CE | eval CE | " +
          " | ".join(f"acc {h}" for h in HELDOUT) + " | agg acc |")
        W("|---|---:|---:|" + "---:|" * (len(HELDOUT) + 1))
        for tag, tce, ece, per, agg in history:
            cells = " | ".join(f"{per[h]:.3f}" for h in HELDOUT)
            W(f"| {tag} | {tce:.4f} | {ece:.4f} | {cells} | {agg:.3f} |")
        base_agg = history[0][4]
        best = max(history[1:], key=lambda r: r[4]) if len(history) > 1 else history[0]
        W(f"\nheld-out aggregate acc: baseline **{base_agg:.3f}** -> best "
          f"**{best[4]:.3f}** (epoch {best[0]}), delta **{best[4]-base_agg:+.3f}**\n")
    W("_See the console log / final message for the written verdict._\n")
    with open(p, "w") as f:
        f.write("\n".join(L))
    log(f"\nwrote {p}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="both", choices=["core", "proj", "both"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    logf = open(os.path.join(OUT, "train_log.txt"), "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(str(msg) + "\n")
        logf.flush()

    log(f"\n##### run {time.strftime('%Y-%m-%d %H:%M:%S')} args={vars(args)} #####")
    scopes = ["core", "proj"] if args.scope == "both" else [args.scope]
    runs = {}
    tstats = estats = None
    for scope in scopes:
        history, tstats, estats = train(scope, args.epochs, args.lr, args.seed, log)
        runs[scope] = (history, tstats, estats)
        mx.clear_cache()
    write_report(runs, tstats, estats, args, log)
    log("DONE")


if __name__ == "__main__":
    main()
