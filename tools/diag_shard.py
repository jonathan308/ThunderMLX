import os, resource, sys
os.environ["MLX_TRUST_REMOTE_CODE"]="true"
os.environ["HF_HUB_OFFLINE"]="1"
import mlx.core as mx
from pathlib import Path

def load_env_file(path):
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

base = Path(__file__).resolve().parents[1]
for env_name in (".env.local", "m3_cluster.env", ".env"):
    load_env_file(base / env_name)

def free_gb():
    import subprocess
    for line in subprocess.run(["vm_stat"],capture_output=True,text=True).stdout.splitlines():
        if "Pages free" in line:
            return int(line.split()[-1].rstrip("."))*16384/1024/1024/1024
    return -1

g = mx.distributed.init()
rank, size = g.rank(), g.size()
print(f"[rank {rank}] init free={free_gb():.1f}GB", flush=True)

from mlx_vlm.utils import load_model
mp = Path(os.environ.get("MLX_M3_MODEL", "mlx-community/MiniMax-M3-4bit"))
model = load_model(mp, lazy=True, strict=False)
print(f"[rank {rank}] lazy load done free={free_gb():.1f}GB", flush=True)

model.language_model.shard(g)
print(f"[rank {rank}] shard done free={free_gb():.1f}GB", flush=True)

# count params this rank would materialize
import mlx.nn as nn
leaves = nn.utils.tree_flatten(model.language_model.parameters())
n = sum(p.size if hasattr(p,'size') else 0 for _,p in leaves)
print(f"[rank {rank}] {len(leaves)} param tensors, ~{n/1e9:.1f}G params to eval", flush=True)

print(f"[rank {rank}] starting eval...", flush=True)
mx.eval(model.language_model.parameters())
print(f"[rank {rank}] EVAL DONE free={free_gb():.1f}GB", flush=True)
mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))
print(f"[rank {rank}] barrier done", flush=True)
