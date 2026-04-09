# -*- coding: utf-8 -*-
"""实验跟踪工具，兼容 TSV 与 TensorBoard 两种记录方式。"""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .tb_logger import TensorBoardLogger


class ExperimentTracker:
    """负责把训练/评估标量写入磁盘，并在可用时同步到 TensorBoard。"""

    def __init__(self, config: dict[str, Any], run_dir: Path | None = None) -> None:
        self.tb_logger = TensorBoardLogger(config)
        self.run_dir = run_dir or self.tb_logger.run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.scalar_path = self.run_dir / "scalars.tsv"

        if not self.scalar_path.exists():
            self.scalar_path.write_text("step\ttag\tvalue\n", encoding="utf-8")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """记录单个标量，便于后续画曲线或做回归比较。"""
        with self.scalar_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{step}\t{tag}\t{value:.8f}\n")

        self.tb_logger.add_scalar(tag, value, step)

    def log_hparams(self, hparams: dict[str, Any], metrics: dict[str, float] | None = None) -> None:
        """记录超参数，便于 TensorBoard HParams 面板比较实验。"""
        self.tb_logger.log_hparams(hparams, metrics)

    def log_model_graph(self, model: Any, input_to_model: Any) -> None:
        """在图可追踪时记录模型结构。"""
        self.tb_logger.add_graph(model, input_to_model)

    def log_parameter_histograms(self, named_parameters: Iterable[tuple[str, Any]], step: int, limit: int = 4) -> None:
        """记录少量关键参数的分布，避免 event 文件过大。"""
        logged = 0
        for name, parameter in named_parameters:
            if logged >= limit:
                break
            if parameter is None or not getattr(parameter, "requires_grad", False):
                continue
            if getattr(parameter, "ndim", 0) < 2:
                continue
            self.tb_logger.add_histogram(f"params/{name}", parameter.detach().cpu(), step)
            logged += 1

    def flush(self) -> None:
        """主动刷盘，减少异常退出时的日志丢失。"""
        self.tb_logger.flush()

    def close(self) -> None:
        """关闭底层 writer，确保缓存内容落盘。"""
        self.tb_logger.close()
