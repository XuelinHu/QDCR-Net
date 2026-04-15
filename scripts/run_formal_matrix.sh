#!/usr/bin/env bash
# 使用 UTF-8 编写注释。
# 该脚本顺序执行正式实验矩阵中的训练与评估任务。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/outputs/automation"
mkdir -p "$LOG_DIR"

GPU_MEM_USED_THRESHOLD="${GPU_MEM_USED_THRESHOLD:-4500}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-180}"

DATASET_FILTER="${DATASET_FILTER:-}"
MODEL_FILTER="${MODEL_FILTER:-}"
CONFIG_GLOB="${CONFIG_GLOB:-configs/formal/*.yaml}"

timestamp() {
  date '+%F %T'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

wait_for_gpu() {
  # 如果机器支持 nvidia-smi，则等待显存占用下降到阈值以下后再启动任务。
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
  # 单个配置先训练再评估，并把日志分别落到自动化输出目录。
  local config_path="$1"
  local stem
  local model_name
  local train_cmd
  local eval_cmd
  stem="$(basename "$config_path" .yaml)"
  model_name="$(python - <<PY
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("$config_path").read_text())
print(str(cfg.get("model", {}).get("name", "qdcr_net")))
PY
)"

  case "$model_name" in
    yolov8|rtdetr|underwater_enhance_yolov8)
      train_cmd=(python scripts/train_ultralytics.py --config "$config_path")
      eval_cmd=(python scripts/eval_ultralytics.py --config "$config_path")
      ;;
    faster_rcnn)
      train_cmd=(python scripts/train_faster_rcnn.py --config "$config_path")
      eval_cmd=(python scripts/eval_faster_rcnn.py --config "$config_path")
      ;;
    *)
      train_cmd=(python scripts/train.py --config "$config_path")
      eval_cmd=(python scripts/eval_map.py --config "$config_path")
      ;;
  esac

  wait_for_gpu
  log "train start: $config_path model=${model_name}"
  conda run -n yolo "${train_cmd[@]}" 2>&1 | tee "$LOG_DIR/${stem}_train.log"

  wait_for_gpu
  log "eval start: $config_path model=${model_name}"
  conda run -n yolo "${eval_cmd[@]}" 2>&1 | tee "$LOG_DIR/${stem}_eval.log"
}

should_run() {
  local config_path="$1"
  local stem
  stem="$(basename "$config_path" .yaml)"
  if [[ -n "$DATASET_FILTER" && ! "$stem" =~ $DATASET_FILTER ]]; then
    return 1
  fi
  if [[ -n "$MODEL_FILTER" && ! "$stem" =~ $MODEL_FILTER ]]; then
    return 1
  fi
  return 0
}

collect_configs() {
  local config_path
  for config_path in $CONFIG_GLOB; do
    [[ -f "$config_path" ]] || continue
    should_run "$config_path" || continue
    printf '%s\n' "$config_path"
  done | sort
}

main() {
  # 默认自动发现 configs/formal/*.yaml，后续新增模型配置后脚本无需再次硬编码。
  # 示例：
  #   MODEL_FILTER='^(base|qdcr)_' bash scripts/run_formal_matrix.sh
  #   DATASET_FILTER='ruod' bash scripts/run_formal_matrix.sh
  #   MODEL_FILTER='^(yolov8|rtdetr)_' DATASET_FILTER='brackish' bash scripts/run_formal_matrix.sh
  mapfile -t configs < <(collect_configs)
  if [[ "${#configs[@]}" -eq 0 ]]; then
    log "no formal configs matched; CONFIG_GLOB=${CONFIG_GLOB} DATASET_FILTER=${DATASET_FILTER:-<empty>} MODEL_FILTER=${MODEL_FILTER:-<empty>}"
    exit 1
  fi

  log "formal matrix runner started"
  log "selected configs (${#configs[@]}): ${configs[*]}"
  for config_path in "${configs[@]}"; do
    run_one "$config_path"
  done
  log "formal matrix runner completed"
}

main "$@"
