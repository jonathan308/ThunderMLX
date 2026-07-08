"""Quick test: does each pipeline rank identify ONLY its own layer files?
Honors M3_PIPELINE_LAYERS env for asymmetric splits. Does NOT load weights."""
import os, sys, json
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["MLX_TRUST_REMOTE_CODE"] = "true"
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

group = mx.distributed.init()
rank, size = group.rank(), group.size()

model_path = Path(os.environ.get("MLX_M3_MODEL", "mlx-community/MiniMax-M3-4bit"))
with open(model_path / "model.safetensors.index.json") as f:
    weight_index = json.load(f)["weight_map"]

# Reproduce the EXACT split logic from m3_pipeline_patch.py (incl. env override)
n_layers = 60
spec = os.environ.get("M3_PIPELINE_LAYERS")
if spec:
    counts = [int(x) for x in spec.split(",")]
    end_from_left = n_layers
    ranges = []
    for r in range(size):
        c = counts[r]
        ranges.append((end_from_left - c, end_from_left))
        end_from_left -= c
    start_idx, end_idx = ranges[rank]
else:
    layers_per_rank = n_layers // size
    extra = n_layers - layers_per_rank * size
    if rank < extra:
        layers_per_rank += 1
    start_idx = (size - rank - 1) * layers_per_rank
    end_idx = start_idx + layers_per_rank

# File sizes
file_sizes = {}
for k, fname in weight_index.items():
    if fname not in file_sizes:
        fp = model_path / fname
        file_sizes[fname] = os.path.getsize(fp) if fp.exists() else 0

# Which files does this rank need?
my_files = set()
for k, fname in weight_index.items():
    if ".layers." in k:
        try:
            li = int(k.split(".layers.")[1].split(".")[0])
        except (ValueError, IndexError):
            continue
        if start_idx <= li < end_idx:
            my_files.add(fname)
    elif "embed_tokens" in k:
        if rank == size - 1:
            my_files.add(fname)
    elif "lm_head" in k:
        if rank == 0:
            my_files.add(fname)

total_bytes = sum(file_sizes.get(f, 0) for f in my_files)
all_files = set(weight_index.values())
print(f"[rank {rank}/{size}] layers [{start_idx}:{end_idx}] = {end_idx-start_idx} layers", flush=True)
print(f"[rank {rank}] needs {len(my_files)}/{len(all_files)} shard files (~{total_bytes/1e9:.0f} GB)", flush=True)
mx.eval(mx.distributed.all_sum(mx.array([float(rank)])))
print(f"[rank {rank}] OK", flush=True)
