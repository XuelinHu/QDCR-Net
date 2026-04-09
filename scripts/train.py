# -*- coding: utf-8 -*-
"""训练入口脚本。"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
# 保证直接执行脚本时也能导入项目源码。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine.trainer import Trainer
from src.utils import get_logger
from src.utils.config import load_config
from src.utils.experiment_notify import send_experiment_notification


def parse_args() -> argparse.Namespace:
    """解析训练命令行参数。"""
    parser = argparse.ArgumentParser(description="Train a detection model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "qdcr_net.yaml",
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    """加载配置并启动训练流程。"""
    logger = get_logger("qdcr.train")
    args = parse_args()
    config = load_config(args.config)
    trainer = Trainer(config)
    start_time = datetime.now()
    try:
        summary = trainer.fit()
        end_time = datetime.now()
        send_experiment_notification(
            config=config,
            project_root=ROOT,
            trainer=trainer,
            start_time=start_time,
            end_time=end_time,
            success=True,
            stage="训练",
            summary=summary,
        )
    except Exception as exc:
        end_time = datetime.now()
        logger.exception("[train] training failed")
        send_experiment_notification(
            config=config,
            project_root=ROOT,
            trainer=trainer,
            start_time=start_time,
            end_time=end_time,
            success=False,
            stage="训练",
            error=exc,
        )
        raise


if __name__ == "__main__":
    main()
