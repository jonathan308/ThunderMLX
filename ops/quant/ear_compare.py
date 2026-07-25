#!/usr/bin/env python3
"""Pairwise EAR between two ear_eval.py captures.

EAR (arXiv 2605.02404): per position, sum of min(p,q) over the union of the
two arms' top-10 token sets; 1.0 = identical next-token distributions.
Their thresholds: >=0.985-0.99 ~ distribution-lossless.

Usage: ear_compare.py a.npz b.npz [--top 10]
"""
import argparse
import json

import numpy as np


def ear(a, b, top=10):
    ids_a, p_a = a["top_ids"], a["top_probs"]
    ids_b, p_b = b["top_ids"], b["top_probs"]
    assert ids_a.shape == ids_b.shape, "captures differ in shape"
    B, T, K = ids_a.shape
    out = np.zeros((B, T), dtype=np.float64)
    for bi in range(B):
        for t in range(T):
            da = dict(zip(ids_a[bi, t, :top].tolist(), p_a[bi, t, :top].tolist()))
            db = dict(zip(ids_b[bi, t, :top].tolist(), p_b[bi, t, :top].tolist()))
            out[bi, t] = sum(min(da.get(k, 0.0), db.get(k, 0.0))
                             for k in set(da) | set(db))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    a, b = np.load(args.a), np.load(args.b)
    ma = json.loads(str(a["meta"])) if "meta" in a else {}
    mb = json.loads(str(b["meta"])) if "meta" in b else {}
    if not np.array_equal(a["ids"], b["ids"]):
        raise SystemExit("FATAL: corpora differ — captures not comparable")
    e = ear(a, b, args.top)
    flat = e.ravel()
    print(f"A: {ma.get('model', args.a)}")
    print(f"B: {mb.get('model', args.b)}")
    print(f"positions: {flat.size}")
    print(f"EAR mean {flat.mean():.4f} | median {np.median(flat):.4f} | "
          f"p5 {np.percentile(flat, 5):.4f} | p1 {np.percentile(flat, 1):.4f} | "
          f"min {flat.min():.4f}")
    print(f"fraction of positions >= 0.99: {(flat >= 0.99).mean():.3f} | "
          f">= 0.985: {(flat >= 0.985).mean():.3f}")
    per_seq = e.mean(axis=1)
    worst = np.argsort(per_seq)[:3]
    print("worst sequences (idx: mean EAR):",
          {int(i): round(float(per_seq[i]), 4) for i in worst})


if __name__ == "__main__":
    main()
