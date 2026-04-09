# -*- coding: utf-8 -*-
"""训练/评估完成后的钉钉通知辅助工具。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .dingtalk_util import send_to_dingtalk
from .logger import get_logger


def infer_dataset_name(config: dict[str, Any]) -> str:
    """从配置中推断数据集名称。"""
    dataset_config = config.get("dataset", {})
    if dataset_config.get("name"):
        return str(dataset_config["name"])
    for key in ("train_root", "val_root"):
        path_value = dataset_config.get(key)
        if not path_value:
            continue
        path = Path(path_value)
        parts = path.parts
        for marker in ("train", "valid", "val"):
            if marker in parts:
                index = parts.index(marker)
                if index > 0:
                    return parts[index - 1]
        if len(parts) >= 2:
            return parts[-2]
    return "dataset"


def fallback_run_name(config: dict[str, Any], start_time: datetime) -> str:
    """在 run_dir 尚未生成时构造一个可区分实验的名称。"""
    experiment_config = config.get("experiment", {})
    model_config = config.get("model", {})
    train_config = config.get("train", {})
    return (
        f"{model_config.get('name', 'model')}_"
        f"{infer_dataset_name(config)}_"
        f"bs{train_config.get('batch_size', 'na')}_"
        f"lr{train_config.get('lr', 'na')}_"
        f"seed{experiment_config.get('seed', 'na')}_"
        f"{start_time.strftime('%Y%m%d-%H%M%S')}"
    )


def format_duration(start_time: datetime, end_time: datetime) -> str:
    """把运行时长格式化为 `HH:MM:SS`。"""
    seconds = int((end_time - start_time).total_seconds())
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def send_experiment_notification(
    *,
    config: dict[str, Any],
    project_root: Path,
    trainer: Any,
    start_time: datetime,
    end_time: datetime,
    success: bool,
    stage: str,
    summary: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
    error: BaseException | None = None,
) -> None:
    """发送训练或评估的 success/failure 钉钉通知。"""
    logger = get_logger("qdcr.notify")
    experiment_config = config.get("experiment", {})
    model_config = config.get("model", {})
    train_config = config.get("train", {})
    run_dir = getattr(trainer, "run_dir", None)
    run_name = run_dir.name if run_dir is not None else fallback_run_name(config, start_time)
    tb_dir = str(run_dir) if run_dir is not None else "N/A"
    dataset_name = infer_dataset_name(config)
    best_ckpt = str(getattr(trainer, "best_checkpoint_path", "N/A"))
    common_lines = [
        f"## {stage}{'完成' if success else '失败'}",
        f"- 项目名：{project_root.name}",
        f"- 实验名：{experiment_config.get('name', 'unknown')}",
        f"- run_name：{run_name}",
        f"- 模型：{model_config.get('name', 'unknown')}",
        f"- 数据集：{dataset_name}",
        f"- batch_size：{train_config.get('batch_size', 'N/A')}",
        f"- learning_rate：{train_config.get('lr', 'N/A')}",
        f"- epochs：{train_config.get('epochs', 'N/A')}",
        f"- seed：{experiment_config.get('seed', 'N/A')}",
        f"- 开始时间：{start_time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- {'结束时间' if success else '失败时间'}：{end_time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 总耗时：{format_duration(start_time, end_time)}",
        f"- TensorBoard：{tb_dir}",
        f"- 最优权重：{best_ckpt}",
    ]
    if success:
        metric_source = metrics or (summary or {}).get("last_val_metrics", {})
        success_lines = [
            f"- loss：{metric_source.get('loss', 'N/A')}",
            f"- acc：{metric_source.get('acc', 'N/A')}",
            f"- mAP：{metric_source.get('map50', 'N/A')}",
            f"- precision：{metric_source.get('precision', 'N/A')}",
            f"- recall：{metric_source.get('recall', 'N/A')}",
        ]
        if summary:
            success_lines.extend(
                [
                    f"- best_epoch：{summary.get('best_epoch', 'N/A')}",
                    f"- best_metric：{summary.get('best_metric', 'N/A')}",
                    f"- monitor：{summary.get('monitor', 'N/A')}",
                ]
            )
        success_lines.append(f"- 说明：{stage}已完成")
        message = "\n".join(common_lines + success_lines)
    else:
        error_name = type(error).__name__ if error is not None else "UnknownError"
        error_message = str(error) if error is not None else "unknown failure"
        failure_lines = [
            f"- 错误摘要：{error_name}: {error_message[:300]}",
            f"- 说明：{stage}执行失败",
        ]
        message = "\n".join(common_lines + failure_lines)

    try:
        send_to_dingtalk(message, err=not success)
        logger.info("[notify] dingtalk %s message sent stage=%s", "success" if success else "failure", stage)
    except Exception as exc:
        logger.warning("[notify] dingtalk message failed stage=%s error=%s", stage, exc)
