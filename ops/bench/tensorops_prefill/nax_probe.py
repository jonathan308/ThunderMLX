"""Tiny cross-rank probe: is the Metal-4 nax (M5 tensor-op) path active?
Reports GPU arch, and bf16 matmul + 4-bit qmm throughput at MoE-ish shapes.
Small (<1GB), brief (<15s). Safe to run alongside production."""
import time, platform, subprocess
import mlx.core as mx

def dev_arch():
    try:
        di = mx.device_info()
    except Exception:
        di = mx.metal.device_info()
    return di.get("architecture", "?")

def macos():
    try:
        return subprocess.check_output(["sw_vers", "-productVersion"]).decode().strip()
    except Exception:
        return platform.mac_ver()[0]

def bench(fn, iters=30, warm=8):
    for _ in range(warm):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / iters

print(f"arch={dev_arch()}  macOS={macos()}  mlx={mx.__version__}")

# bf16 dense matmul: [M,K] @ [K,N]
for (Mr, K, N) in [(4096, 6144, 6144), (2048, 6144, 8192)]:
    a = mx.random.normal((Mr, K)).astype(mx.bfloat16)
    b = mx.random.normal((K, N)).astype(mx.bfloat16)
    mx.eval(a, b)
    t = bench(lambda: a @ b)
    tflops = 2 * Mr * K * N / t / 1e12
    print(f"  bf16 matmul [{Mr}x{K}]@[{K}x{N}]  {t*1e3:7.3f} ms  {tflops:6.1f} TFLOP/s")

# 4-bit quantized_matmul (transpose=True, K%64==0) -> nax-eligible large-M path
for (Mr, K, N) in [(4096, 6144, 6144), (2048, 6144, 8192)]:
    w = mx.random.normal((N, K)).astype(mx.bfloat16)
    wq, s, bz = mx.quantize(w, group_size=64, bits=4)
    x = mx.random.normal((Mr, K)).astype(mx.bfloat16)
    mx.eval(wq, s, bz, x)
    t = bench(lambda: mx.quantized_matmul(x, wq, s, bz, transpose=True, group_size=64, bits=4))
    tflops = 2 * Mr * K * N / t / 1e12
    print(f"  4bit qmm    [{Mr}x{K}]@[{K}x{N}]  {t*1e3:7.3f} ms  {tflops:6.1f} TFLOP/s")

# --- NAX correctness landmine (PR #3593): M>8 GPU matmul must match a reference ---
# Compare (a) large-M GPU bf16 matmul vs (b) the SAME rows done M=1 at a time
# (M=1 uses the non-nax vector path) and vs (c) an fp32 CPU reference.
K, N, Mbig = 2048, 2048, 64
a = mx.random.normal((Mbig, K)).astype(mx.bfloat16)
b = mx.random.normal((K, N)).astype(mx.bfloat16)
mx.eval(a, b)
gpu_big = (a @ b).astype(mx.float32); mx.eval(gpu_big)          # M>8 -> nax on M5
rows = [ (a[i:i+1] @ b).astype(mx.float32) for i in range(Mbig) ]
gpu_rows = mx.concatenate(rows, axis=0); mx.eval(gpu_rows)       # M=1 -> vector path
with mx.stream(mx.cpu):
    cpu_ref = (a.astype(mx.float32) @ b.astype(mx.float32)); mx.eval(cpu_ref)
den = mx.maximum(mx.abs(cpu_ref), 1e-3)
err_big = float(mx.max(mx.abs(gpu_big - cpu_ref) / den))
err_rows = float(mx.max(mx.abs(gpu_rows - cpu_ref) / den))
err_bigrow = float(mx.max(mx.abs(gpu_big - gpu_rows)))
print(f"  NAX-correctness M={Mbig}: relerr(bigM vs cpu_fp32)={err_big:.4f}  "
      f"relerr(M=1 vs cpu)={err_rows:.4f}  maxabs(bigM vs M=1)={err_bigrow:.4f}")
verdict = "OK" if err_big < 0.05 else "*** SUSPECT NAX MISCOMPILE (M>8 garbage) ***"
print(f"  NAX-correctness verdict: {verdict}")
