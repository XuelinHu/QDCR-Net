#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/outputs/automation"
mkdir -p "$LOG_DIR"

CONFIGS=(
  "configs/formal/base_ruod_testb_eval.yaml"
  "configs/formal/base_ruod_testc_eval.yaml"
  "configs/formal/base_ruod_testl_eval.yaml"
  "configs/formal/qdcr_ruod_testb_eval.yaml"
  "configs/formal/qdcr_ruod_testc_eval.yaml"
  "configs/formal/qdcr_ruod_testl_eval.yaml"
)

timestamp() {
  date '+%F %T'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

main() {
  log "ruod degradation evaluation started"
  for config_path in "${CONFIGS[@]}"; do
    stem="$(basename "$config_path" .yaml)"
    log "eval start: $config_path"
    conda run -n yolo python scripts/eval_map.py --config "$config_path" 2>&1 | tee "$LOG_DIR/${stem}.log"
  done
  log "ruod degradation evaluation completed"
}

main "$@"
