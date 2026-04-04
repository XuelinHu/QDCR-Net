#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p data/raw data/datasets/downloads data/logs configs/generated
LOG="data/logs/download_prepare.log"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"
}

download_extract_generate() {
  local name="$1"
  local url="$2"
  local qdcr_cfg="$3"
  local base_cfg="$4"
  log "Downloading ${name}"
  curl -L -C - -o "data/raw/${name}.zip" "$url" >>"$LOG" 2>&1
  log "Extracting ${name}"
  unzip -o "data/raw/${name}.zip" -d "data/datasets/downloads/${name}" >>"$LOG" 2>&1
  log "Generating configs for ${name}"
  conda run -n yolo python scripts/generate_sample_config.py \
    --dataset-root "data/datasets/downloads/${name}" \
    --template "configs/qdcr_net.yaml" \
    --output "${qdcr_cfg}" \
    --experiment-name "$(basename "${qdcr_cfg}" .yaml)" >>"$LOG" 2>&1
  conda run -n yolo python scripts/generate_sample_config.py \
    --dataset-root "data/datasets/downloads/${name}" \
    --template "configs/base.yaml" \
    --output "${base_cfg}" \
    --experiment-name "$(basename "${base_cfg}" .yaml)" >>"$LOG" 2>&1
  log "Finished ${name}"
}

download_extract_generate "Brackish" "https://ndownloader.figshare.com/files/53414546" "configs/generated/qdcr_brackish_sample.yaml" "configs/generated/base_brackish_sample.yaml"
download_extract_generate "RUOD" "https://ndownloader.figshare.com/files/53414735" "configs/generated/qdcr_ruod_sample.yaml" "configs/generated/base_ruod_sample.yaml"
