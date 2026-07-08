#!/bin/zsh
# Two-stage MLX wheel build (parity recipe from HANDOFF-2026-07-05).
# NEVER single-stage: a bare `pip wheel .` produces a hybrid with corrupt Metal
# kernels (token salad at normal t/s). Stage 1 = mlx, stage 2 = mlx-metal,
# `setup.py clean --all` between, ARCHFLAGS=-arch arm64, both wheels
# version-matched from the same checkout.
#
# Usage: build_mlx_variant.sh <git-ref> <label> [src_dir]
#   e.g. build_mlx_variant.sh origin/main            upstream-main
#        build_mlx_variant.sh exofork/address-rdma-gpu-locks exo-fork
set -e
REF=$1
LABEL=$2
SRC=${3:-$HOME/mlx-src}
OUT=$HOME/minimax-m3-cluster/runtime_patches/variants/$LABEL
[[ -z "$REF" || -z "$LABEL" ]] && { echo "usage: build_mlx_variant.sh <ref> <label> [src]"; exit 2 }
[[ -d $SRC/.git ]] || { echo "no git checkout at $SRC"; exit 2 }

echo "=== building mlx variant '$LABEL' from $REF ==="
cd $SRC
git fetch --all --quiet
git checkout --quiet "$REF"
git log --oneline -1
python3 setup.py clean --all >/dev/null 2>&1 || true
rm -rf dist build

echo "--- stage 1 (mlx) ---"
ARCHFLAGS="-arch arm64" MLX_BUILD_STAGE=1 python3 -m build -w
echo "--- clean between stages ---"
python3 setup.py clean --all >/dev/null 2>&1 || true
echo "--- stage 2 (mlx-metal) ---"
ARCHFLAGS="-arch arm64" MLX_BUILD_STAGE=2 python3 -m build -w

mkdir -p $OUT
cp dist/*.whl $OUT/
echo "=== wheels for '$LABEL': ==="
ls -la $OUT/*.whl
# both wheels must carry the same version string (mlx is cp314, mlx_metal is
# py3-none — anchor on the version field itself, not the abi tag)
vers=$(ls $OUT/*.whl | sed -E 's/.*-([0-9]+\.[0-9]+\.[0-9][^-]*)-(cp|py)[0-9].*/\1/' | sort -u | wc -l | tr -d ' ')
if [[ "$vers" != "1" ]]; then
  echo "WARNING: wheel versions differ — do NOT install this pair"; exit 1
fi
echo "version-matched OK"
