#!/bin/zsh
# Transactional two-rank runtime updater used by the dashboard.
#
# MLX-VLM, MLX-LM, and the Transformers version required by MLX-VLM are staged
# as exact PyPI wheels and installed with --no-deps so they cannot replace the
# custom MLX core. MLX itself is installed only from a validated,
# version-matched mlx + mlx-metal wheel pair under
# runtime_patches/variants/<label>.
set -euo pipefail
setopt NULL_GLOB

SCRIPT_DIR="${0:A:h}"
CLUSTER="${M3_CLUSTER_DIR:-${SCRIPT_DIR:h}}"
PACKAGE="${1:-}"
RESTART="${2:-1}"
VARIANT="${3:-}"
DRY_RUN="${M3_RUNTIME_UPDATE_DRY_RUN:-0}"

if [[ -f "$CLUSTER/.env.local" ]]; then
  source "$CLUSTER/.env.local"
elif [[ -f "$CLUSTER/m3_cluster.env" ]]; then
  source "$CLUSTER/m3_cluster.env"
elif [[ -f "$CLUSTER/.env" ]]; then
  source "$CLUSTER/.env"
fi

case "$PACKAGE" in
  mlx|mlx-lm|mlx-vlm) ;;
  *) echo "unsupported runtime package: $PACKAGE" >&2; exit 2 ;;
esac
case "$RESTART" in
  0|1) ;;
  *) echo "restart must be 0 or 1" >&2; exit 2 ;;
esac

PY="$CLUSTER/bin/mlx-python"
[[ -x "$PY" ]] || { echo "runtime launcher missing: $PY" >&2; exit 2; }

DIRECT_PEER="${M3_DIRECT_PEER:-${M3_RANK1_DIRECT_SSH:-}}"
FALLBACK_PEER="${M3_RANK1_FALLBACK_SSH:-${M3_TAILSCALE_PEER:-}}"
PEER="${M3_PEER:-}"
if [[ -z "$PEER" ]]; then
  if [[ -n "$DIRECT_PEER" ]] && ssh -o BatchMode=yes -o ConnectTimeout=5 "$DIRECT_PEER" true >/dev/null 2>&1; then
    PEER="$DIRECT_PEER"
  else
    PEER="$FALLBACK_PEER"
  fi
fi
[[ -n "$PEER" ]] || { echo "rank 1 is not configured" >&2; exit 2; }

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=12)
SCP_OPTS=(-q -o BatchMode=yes -o ConnectTimeout=12)
if [[ -n "${M3_SSH_KEY:-}" ]]; then
  SSH_OPTS+=(-i "$M3_SSH_KEY")
  SCP_OPTS+=(-i "$M3_SSH_KEY")
fi

remote() {
  ssh "${SSH_OPTS[@]}" "$PEER" "$1"
}

remote_copy() {
  scp "${SCP_OPTS[@]}" "$1" "$PEER:$2"
}

pkg_version() {
  "$PY" - "$1" <<'PY'
import importlib.metadata as md
import sys
try:
    print(md.version(sys.argv[1]))
except Exception:
    print("")
PY
}

remote_pkg_version() {
  local code="import importlib.metadata as md; print(md.version('$1'))"
  remote "cd ${CLUSTER:q} && ./bin/mlx-python -c ${code:q}"
}

pypi_latest() {
  "$PY" - "$1" <<'PY'
import json
import ssl
import sys
import urllib.request

context = None
try:
    import certifi
    context = ssl.create_default_context(cafile=certifi.where())
except Exception:
    pass
with urllib.request.urlopen(
    f"https://pypi.org/pypi/{sys.argv[1]}/json", timeout=20, context=context
) as response:
    print(json.load(response)["info"]["version"])
PY
}

wheel_version() {
  "$PY" - "$1" <<'PY'
from pathlib import Path
import sys
parts = Path(sys.argv[1]).name.split("-")
if len(parts) < 2:
    raise SystemExit("invalid wheel name")
print(parts[1])
PY
}

LOCK_DIR="/private/tmp/thundermlx-runtime-update.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "another runtime update is already active" >&2
  exit 3
fi
STAGE="/private/tmp/thundermlx-runtime-update-$$"
REMOTE_STAGE="/private/tmp/thundermlx-runtime-update-$$"
mkdir -p "$STAGE/new" "$STAGE/old"
remote "rm -rf ${REMOTE_STAGE:q}; mkdir -p ${REMOTE_STAGE:q}"

STOPPED=0
UPDATED=0
OLD_ARTIFACTS=()
NEW_ARTIFACTS=()

cleanup() {
  rm -rf "$STAGE" "$LOCK_DIR"
  remote "rm -rf ${REMOTE_STAGE:q}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

install_pair() {
  local -a artifacts=("$@")
  "$PY" -m pip install --no-deps --force-reinstall "${artifacts[@]}"
  remote "cd ${CLUSTER:q} && ./bin/mlx-python -m pip install --no-deps --force-reinstall ${REMOTE_STAGE:q}/*.whl"
}

copy_artifacts_to_worker() {
  remote "find ${REMOTE_STAGE:q} -maxdepth 1 -type f -name '*.whl' -delete"
  local artifact
  for artifact in "$@"; do
    remote_copy "$artifact" "$REMOTE_STAGE/"
  done
}

restart_cluster() {
  [[ "$RESTART" == "1" ]] || return 0
  echo "[runtime-update] restarting the managed cluster"
  /bin/zsh "$CLUSTER/M3_Start.command"
}

rollback() {
  [[ "$UPDATED" == "1" && ${#OLD_ARTIFACTS[@]} -gt 0 ]] || return 0
  echo "[runtime-update] validation failed; restoring previous artifacts" >&2
  copy_artifacts_to_worker "${OLD_ARTIFACTS[@]}" || true
  install_pair "${OLD_ARTIFACTS[@]}" || true
}

on_error() {
  local rc=$?
  trap - ERR
  rollback
  if [[ "$STOPPED" == "1" ]]; then
    restart_cluster || true
  fi
  echo "RUNTIME_UPDATE_RESULT failed package=$PACKAGE code=$rc" >&2
  exit "$rc"
}
trap on_error ERR

echo "[runtime-update] package=$PACKAGE rank0=$PY rank1=$PEER:$CLUSTER/bin/mlx-python"
remote "cd ${CLUSTER:q} && test -x ./bin/mlx-python && ./bin/mlx-python -c 'import sys; print(sys.executable)'"

if [[ "$PACKAGE" == "mlx" ]]; then
  approved_variant="$($PY - "$CLUSTER/runtime_patches/mlx_variants.json" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1]))
label = str(manifest.get("recommended") or "")
record = (manifest.get("variants") or {}).get(label) or {}
if str(record.get("status") or "").lower() not in {"production", "validated"}:
    raise SystemExit("recommended MLX variant is not production-validated")
print(label)
PY
)"
  if [[ -z "$VARIANT" ]]; then
    VARIANT="$approved_variant"
  fi
  if [[ "$VARIANT" != "$approved_variant" && "${M3_RUNTIME_ALLOW_UNVALIDATED_MLX:-0}" != "1" ]]; then
    echo "MLX variant $VARIANT is not the approved production pair $approved_variant" >&2
    exit 2
  fi
  [[ "$VARIANT" =~ '^[A-Za-z0-9_.-]+$' ]] || { echo "invalid MLX variant: $VARIANT" >&2; exit 2; }
  VARIANT_DIR="$CLUSTER/runtime_patches/variants/$VARIANT"
  NEW_ARTIFACTS=("$VARIANT_DIR"/mlx-[0-9]*.whl "$VARIANT_DIR"/mlx_metal-[0-9]*.whl)
  [[ ${#NEW_ARTIFACTS[@]} -eq 2 && -f "${NEW_ARTIFACTS[1]}" && -f "${NEW_ARTIFACTS[2]}" ]] || {
    echo "validated MLX wheel pair is missing for variant $VARIANT" >&2
    exit 2
  }
  target_mlx="$(wheel_version "${NEW_ARTIFACTS[1]}")"
  target_metal="$(wheel_version "${NEW_ARTIFACTS[2]}")"
  [[ "$target_mlx" == "$target_metal" ]] || { echo "MLX wheel versions do not match" >&2; exit 2; }
  current="$(pkg_version mlx)"
  for candidate in "$CLUSTER"/runtime_patches/variants/*/mlx-"$current"-*.whl; do
    [[ -f "$candidate" ]] || continue
    metals=("${candidate:h}"/mlx_metal-"$current"-*.whl)
    if [[ ${#metals[@]} -eq 1 && -f "${metals[1]}" ]]; then
      OLD_ARTIFACTS=("$candidate" "${metals[1]}")
      break
    fi
  done
  [[ ${#OLD_ARTIFACTS[@]} -eq 2 ]] || { echo "rollback wheel pair is missing for installed MLX $current" >&2; exit 2; }
  echo "[runtime-update] MLX $current -> $target_mlx via validated variant $VARIANT"
else
  UPDATE_PACKAGES=("$PACKAGE")
  if [[ "$PACKAGE" == "mlx-vlm" ]]; then
    UPDATE_PACKAGES+=("mlx-lm" "transformers")
  fi
  primary_current=""
  for staged_package in "${UPDATE_PACKAGES[@]}"; do
    current="$(pkg_version "$staged_package")"
    if [[ "$staged_package" == "$PACKAGE" && -n "${M3_RUNTIME_UPDATE_TARGET:-}" ]]; then
      target="$M3_RUNTIME_UPDATE_TARGET"
    else
      target="$(pypi_latest "$staged_package")"
    fi
    [[ -n "$current" && -n "$target" ]] || { echo "could not resolve $staged_package versions" >&2; exit 2; }
    [[ -n "$primary_current" ]] || primary_current="$current"
    echo "[runtime-update] $staged_package $current -> $target"
    "$PY" -m pip download --disable-pip-version-check --no-deps --only-binary=:all: \
      --dest "$STAGE/new" "$staged_package==$target"
    "$PY" -m pip download --disable-pip-version-check --no-deps --only-binary=:all: \
      --dest "$STAGE/old" "$staged_package==$current"
  done
  current="$primary_current"
  NEW_ARTIFACTS=("$STAGE/new"/*.whl)
  OLD_ARTIFACTS=("$STAGE/old"/*.whl)
  [[ ${#NEW_ARTIFACTS[@]} -eq ${#UPDATE_PACKAGES[@]} && ${#OLD_ARTIFACTS[@]} -eq ${#UPDATE_PACKAGES[@]} ]] || {
    echo "staged wheel count does not match the runtime package set" >&2
    exit 2
  }
fi

copy_artifacts_to_worker "${NEW_ARTIFACTS[@]}"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "RUNTIME_UPDATE_RESULT dry_run_ok package=$PACKAGE worker=$PEER restart=$RESTART"
  exit 0
fi

echo "[runtime-update] syncing source and runtime patchers to rank 1"
/bin/zsh "$CLUSTER/sync_rank1.sh"
echo "[runtime-update] stopping inference ranks before changing imported packages"
M3_STOP_KEEP_DASHBOARD=1 M3_STOP_KEEP_GATEWAY=1 M3_STOP_DRAIN_SECONDS=120 \
  /bin/zsh "$CLUSTER/stop_cluster.sh"
STOPPED=1

UPDATED=1
install_pair "${NEW_ARTIFACTS[@]}"

if [[ "$PACKAGE" == "mlx-vlm" ]]; then
  for patcher in \
      apply_mlx_vlm_prefill_progress_patch.py \
      apply_mlx_vlm_prefill_clear_cache_patch.py; do
    if [[ -f "$CLUSTER/runtime_patches/$patcher" ]]; then
      "$PY" "$CLUSTER/runtime_patches/$patcher"
      remote "cd ${CLUSTER:q} && ./bin/mlx-python runtime_patches/${patcher:q}"
    fi
  done
fi

VERIFY_PACKAGES=("$PACKAGE")
if [[ "$PACKAGE" == "mlx-vlm" ]]; then
  VERIFY_PACKAGES+=("mlx-lm" "transformers")
fi
local_version=""
for verify_package in "${VERIFY_PACKAGES[@]}"; do
  rank0_version="$(pkg_version "$verify_package")"
  rank1_version="$(remote_pkg_version "$verify_package")"
  [[ -n "$rank0_version" && "$rank0_version" == "$rank1_version" ]] || {
    echo "$verify_package rank version mismatch: rank0=$rank0_version rank1=$rank1_version" >&2
    exit 1
  }
  [[ -n "$local_version" ]] || local_version="$rank0_version"
done
"$PY" "$CLUSTER/ops/check_runtime_compat.py" \
  "$CLUSTER/runtime_patches/mlx_variants.json"
remote "cd ${CLUSTER:q} && ./bin/mlx-python ops/check_runtime_compat.py runtime_patches/mlx_variants.json"

if [[ "$PACKAGE" == "mlx" ]]; then
  "$PY" "$CLUSTER/ops/known_answer.py"
  remote "cd ${CLUSTER:q} && ./bin/mlx-python ops/known_answer.py"
fi

restart_cluster
STOPPED=0
UPDATED=0
echo "RUNTIME_UPDATE_RESULT ok package=$PACKAGE version=$local_version worker=$PEER restart=$RESTART"
