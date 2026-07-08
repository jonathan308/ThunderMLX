#!/usr/bin/env python3
"""Kernel known-answer gate for custom MLX builds (recreated 2026-07-05; the
original lived in /tmp and was lost to a reboot).

Catches the corrupt-Metal-kernel failure mode of single-stage builds (token
salad at normal t/s): every test computes the same op on the GPU stream and the
CPU stream and requires agreement. The 4-bit quantized matmul path is the one
MiniMax-M3-4bit actually exercises — it is the test that matters most.

Run on BOTH ranks after any wheel swap, BEFORE the coherence gate, BEFORE any
soak. Exit 0 = pass.
"""
import sys

import mlx.core as mx


def close(a, b, rtol, name, report):
    """Relative agreement: corrupt kernels produce O(1) relative garbage;
    legit GPU-vs-CPU accumulation-order differences sit orders below these
    thresholds. Absolute tolerances proved too tight across GPU generations
    (rank1 M-series accumulates differently than rank0 at fp32)."""
    a32, b32 = a.astype(mx.float32), b.astype(mx.float32)
    max_diff = mx.max(mx.abs(a32 - b32)).item()
    ref = max(mx.max(mx.abs(b32)).item(), 1e-6)
    rel = max_diff / ref
    report.append(f"    {name}: max_abs_diff={max_diff:.3e} rel={rel:.3e} (rtol {rtol})")
    return rel < rtol


def main() -> int:
    mx.random.seed(7)
    failures = []
    report = []

    # 1. plain matmul fp32: gpu vs cpu (rtol generous: catches garbage, not rounding)
    a = mx.random.normal((256, 512))
    b = mx.random.normal((512, 128))
    g = mx.matmul(a, b, stream=mx.gpu)
    c = mx.matmul(a, b, stream=mx.cpu)
    mx.eval(g, c)
    if not close(g, c, 1e-3, "matmul_fp32", report):
        failures.append("matmul_fp32 gpu/cpu divergence")

    # 2. 4-bit quantized matmul (the MiniMax-4bit hot path)
    # transpose=True (default): w is (out, in), computes x @ w.T
    w = mx.random.normal((1024, 512))
    wq, scales, biases = mx.quantize(w, bits=4)
    x = mx.random.normal((8, 512))
    g = mx.quantized_matmul(x, wq, scales, biases, bits=4, stream=mx.gpu)
    c = mx.quantized_matmul(x, wq, scales, biases, bits=4, stream=mx.cpu)
    mx.eval(g, c)
    if not close(g, c, 1e-2, "qmm_4bit", report):
        failures.append("quantized_matmul_4bit gpu/cpu divergence")

    # 3. softmax -> argmax chain (decode head path) — exact index agreement
    logits = mx.random.normal((4, 32000))
    g = mx.argmax(mx.softmax(logits, axis=-1), axis=-1)
    c = mx.argmax(mx.softmax(logits.astype(mx.float32), axis=-1, stream=mx.cpu), axis=-1, stream=mx.cpu)
    mx.eval(g, c)
    agree = int(mx.sum(g == c).item())
    report.append(f"    softmax/argmax: {agree}/4 rows agree")
    if agree < 4:
        failures.append("softmax/argmax gpu/cpu divergence")

    # 4. fp16 layernorm-ish chain (norm + scale + residual)
    h = mx.random.normal((64, 2048)).astype(mx.float16)
    def norm_chain(t, stream):
        m = mx.mean(t, axis=-1, keepdims=True, stream=stream)
        v = mx.var(t, axis=-1, keepdims=True, stream=stream)
        return ((t - m) * mx.rsqrt(v + 1e-5, stream=stream) + t).astype(mx.float32)
    g = norm_chain(h, mx.gpu)
    c = norm_chain(h.astype(mx.float32), mx.cpu)
    mx.eval(g, c)
    if not close(g, c, 5e-2, "norm_chain_fp16", report):
        failures.append("norm_chain fp16 gpu/cpu divergence")

    print(f"mlx {mx.__version__} known-answer: {'FAIL' if failures else 'PASS'}")
    for line in report:
        print(line)
    for f in failures:
        print(f"  FAILED: {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
