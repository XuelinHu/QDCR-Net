# -*- coding: utf-8 -*-
"""训练入口脚本。"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
# 保证直接执行脚本时也能导入项目源码。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine.trainer import Trainer
from src.utils.config import load_config


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
    args = parse_args()
    config = load_config(args.config)
    trainer = Trainer(config)
    trainer.fit()


if __name__ == "__main__":
    main()
