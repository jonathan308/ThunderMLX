import mlx.core as mx
g = mx.distributed.init()
print(f"RANK {g.rank()}/{g.size()} ONLINE", flush=True)
# collective op to prove the two ranks can talk
s = mx.distributed.all_sum(mx.array([float(g.rank())]))
mx.eval(s)
print(f"RANK {g.rank()} collective OK (sum={s.tolist()})", flush=True)
