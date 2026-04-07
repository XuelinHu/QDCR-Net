#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/outputs/automation"
mkdir -p "$LOG_DIR"

GPU_MEM_USED_THRESHOLD="${GPU_MEM_USED_THRESHOLD:-4500}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-180}"

CONFIGS=(
  "configs/formal/base_uov2_full.yaml"
  "configs/formal/qdcr_uov2_full.yaml"
  "configs/formal/base_brackish_full.yaml"
  "configs/formal/qdcr_brackish_full.yaml"
  "configs/formal/base_ruod_full.yaml"
  "configs/formal/qdcr_ruod_full.yaml"
)

timestamp() {
  date '+%F %T'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

wait_for_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi

  while true; do
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
    if [[ -n "$used" && "$used" -le "$GPU_MEM_USED_THRESHOLD" ]]; then
      log "gpu ready: memory.used=${used}MiB"
      return 0
    fi
    log "gpu busy: memory.used=${used:-unknown}MiB, sleep ${GPU_POLL_SECONDS}s"
    sleep "$GPU_POLL_SECONDS"
  done
}

run_one() {
  local config_path="$1"
  local stem
  stem="$(basename "$config_path" .yaml)"

  wait_for_gpu
  log "train start: $config_path"
  conda run -n yolo python scripts/train.py --config "$config_path" 2>&1 | tee "$LOG_DIR/${stem}_train.log"

  wait_for_gpu
  log "eval start: $config_path"
  conda run -n yolo python scripts/eval_map.py --config "$config_path" 2>&1 | tee "$LOG_DIR/${stem}_eval.log"
}

main() {
  log "formal matrix runner started"
  for config_path in "${CONFIGS[@]}"; do
    run_one "$config_path"
  done
  log "formal matrix runner completed"
}

main "$@"
