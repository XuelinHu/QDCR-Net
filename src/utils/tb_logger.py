# -*- coding: utf-8 -*-
"""TensorBoard 日志目录与 writer 封装。"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


class TensorBoardLogger:
    """负责构建独立实验目录并代理 TensorBoard 写入。"""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.project_root = Path(__file__).resolve().parents[2]
        self.run_dir = self._build_run_dir()
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.hparams_path = self.run_dir / "hparams.json"
        self.writer = SummaryWriter(log_dir=str(self.run_dir)) if SummaryWriter is not None else None
        print(f"TensorBoard log dir: {self.run_dir}")

    def _build_run_dir(self) -> Path:
        root_dir = self.project_root / "runs"
        root_dir.mkdir(parents=True, exist_ok=True)
        run_name = self._run_name()
        run_dir = root_dir / run_name
        if not run_dir.exists():
            return run_dir
        suffix = 1
        while True:
            candidate = root_dir / f"{run_name}_{suffix}"
            if not candidate.exists():
                return candidate
            suffix += 1

    def _run_name(self) -> str:
        model_config = self.config.get("model", {})
        dataset_config = self.config.get("dataset", {})
        train_config = self.config.get("train", {})
        experiment_config = self.config.get("experiment", {})
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        parts = [
            timestamp,
            self._sanitize(str(model_config.get("name", experiment_config.get("name", "model")))),
            self._sanitize(self._dataset_name()),
            f"bs{int(train_config.get('batch_size', 0))}",
            f"lr{train_config.get('lr', 'na')}",
        ]
        seed = experiment_config.get("seed")
        if seed is not None:
            parts.append(f"seed{seed}")
        tag = experiment_config.get("name")
        if tag:
            parts.append(self._sanitize(str(tag)))
        return "_".join(part for part in parts if part)

    def _dataset_name(self) -> str:
        dataset_config = self.config.get("dataset", {})
        dataset_name = dataset_config.get("name")
        if dataset_name:
            return str(dataset_name)

        for key in ("train_root", "val_root"):
            root_value = dataset_config.get(key)
            if not root_value:
                continue
            root = Path(root_value)
            parts = [part for part in root.parts if part not in {"", "/", "data", "datasets", "downloads"}]
            if "train" in parts:
                return parts[parts.index("train") - 1]
            if "valid" in parts:
                return parts[parts.index("valid") - 1]
            if "val" in parts:
                return parts[parts.index("val") - 1]
            if len(parts) >= 2:
                return parts[-2]
        return "dataset"

    def _sanitize(self, value: str) -> str:
        sanitized = []
        for char in value:
            if char.isalnum() or char in {"-", "_", "."}:
                sanitized.append(char)
            else:
                sanitized.append("_")
        return "".join(sanitized).strip("_") or "na"

    def log_hparams(self, hparams: dict[str, Any], metrics: dict[str, float] | None = None) -> None:
        serialized = {key: self._serialize(value) for key, value in hparams.items()}
        self.hparams_path.write_text(
            json.dumps(serialized, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if self.writer is None:
            return
        if metrics:
            self.writer.add_hparams(serialized, metrics)
            return
        self.writer.add_text("hparams", json.dumps(serialized, indent=2, ensure_ascii=False), 0)

    def add_graph(self, model: Any, input_to_model: Any) -> None:
        if self.writer is None:
            return
        self.writer.add_graph(model, input_to_model=input_to_model)

    def add_histogram(self, tag: str, values: Any, step: int) -> None:
        if self.writer is None:
            return
        self.writer.add_histogram(tag, values, step)

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer is None:
            return
        self.writer.add_scalar(tag, value, step)

    def flush(self) -> None:
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

    def _serialize(self, value: Any) -> str | bool | int | float:
        if isinstance(value, (bool, int, float, str)):
            return value
        if value is None:
            return "none"
        return str(value)
