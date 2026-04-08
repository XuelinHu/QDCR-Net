#!/usr/bin/env bash
# 使用 UTF-8 编写注释。
# 该脚本用于夜间续跑样例实验，并在 RUOD 压缩包准备好后自动生成配置。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/outputs/automation"
mkdir -p "$LOG_DIR"

GPU_UTIL_THRESHOLD="${GPU_UTIL_THRESHOLD:-20}"
GPU_MEM_USED_THRESHOLD="${GPU_MEM_USED_THRESHOLD:-2048}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-300}"
RUOD_POLL_SECONDS="${RUOD_POLL_SECONDS:-300}"

timestamp() {
  date '+%F %T'
}

log() {
  # 所有信息统一追加到标准输出，并由 tee 写入对应日志。
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

run_logged() {
  # 包装任意命令，自动记录任务名和输出日志。
  local name="$1"
  shift
  log "running: $name"
  "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
}

wait_for_gpu() {
  # 同时检查 GPU 利用率与显存占用，避免和其他任务抢卡。
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "nvidia-smi not found, skip GPU wait"
    return 0
  fi

  while true; do
    local query
    query="$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -n1)"
    local util
    local mem
    util="$(printf '%s' "$query" | cut -d',' -f1 | tr -d ' ')"
    mem="$(printf '%s' "$query" | cut -d',' -f2 | tr -d ' ')"

    if [[ -n "$util" && -n "$mem" && "$util" -le "$GPU_UTIL_THRESHOLD" && "$mem" -le "$GPU_MEM_USED_THRESHOLD" ]]; then
      log "gpu is available: util=${util}% mem=${mem}MiB"
      return 0
    fi

    log "gpu busy: util=${util:-unknown}% mem=${mem:-unknown}MiB, sleep ${GPU_POLL_SECONDS}s"
    sleep "$GPU_POLL_SECONDS"
  done
}

wait_for_valid_zip() {
  # 持续轮询 RUOD 压缩包，直到文件存在且 unzip 校验通过。
  local zip_path="$1"
  while true; do
    if [[ -s "$zip_path" ]] && unzip -tqq "$zip_path" >/dev/null 2>&1; then
      log "zip ready: $zip_path"
      return 0
    fi
    log "zip not ready: $zip_path, sleep ${RUOD_POLL_SECONDS}s"
    sleep "$RUOD_POLL_SECONDS"
  done
}

prepare_ruod() {
  # RUOD 数据集到位后，自动解压并生成 QDCR/Base 两套样例配置。
  local zip_path="$ROOT/data/raw/RUOD.zip"
  local extract_dir="$ROOT/data/datasets/downloads/RUOD"

  wait_for_valid_zip "$zip_path"
  mkdir -p "$extract_dir"

  if [[ ! -d "$extract_dir/train" && ! -d "$extract_dir/valid" && ! -d "$extract_dir/val" ]]; then
    run_logged "ruod_unzip" unzip -o "$zip_path" -d "$extract_dir"
  else
    log "skip unzip, dataset folders already exist in $extract_dir"
  fi

  run_logged \
    "ruod_qdcr_config" \
    conda run -n yolo python scripts/generate_sample_config.py \
    --dataset-root "$extract_dir" \
    --template configs/qdcr_net.yaml \
    --output configs/generated/qdcr_ruod_sample.yaml \
    --experiment-name qdcr_ruod_sample \
    --epochs 3 \
    --batch-size 4 \
    --max-batches 8 \
    --image-size 320 \
    --max-objects 8

  run_logged \
    "ruod_base_config" \
    conda run -n yolo python scripts/generate_sample_config.py \
    --dataset-root "$extract_dir" \
    --template configs/base.yaml \
    --output configs/generated/base_ruod_sample.yaml \
    --experiment-name baseline_ruod_sample \
    --epochs 3 \
    --batch-size 4 \
    --max-batches 8 \
    --image-size 320 \
    --max-objects 8
}

run_sample_suite() {
  # 样例实验统一遵循“训练 -> 评估”顺序。
  local config_path="$1"
  local stem="$2"
  wait_for_gpu
  run_logged "${stem}_train" conda run -n yolo python scripts/train.py --config "$config_path"
  wait_for_gpu
  run_logged "${stem}_eval" conda run -n yolo python scripts/eval_map.py --config "$config_path"
}

main() {
  # 先继续现有 Brackish 样例，再在 RUOD 数据准备好后补跑 RUOD。
  log "overnight continuation started"

  run_sample_suite "configs/generated/qdcr_brackish_sample.yaml" "qdcr_brackish_sample"
  run_sample_suite "configs/generated/base_brackish_sample.yaml" "base_brackish_sample"

  prepare_ruod
  run_sample_suite "configs/generated/qdcr_ruod_sample.yaml" "qdcr_ruod_sample"
  run_sample_suite "configs/generated/base_ruod_sample.yaml" "base_ruod_sample"

  log "overnight continuation completed"
}

main "$@"
