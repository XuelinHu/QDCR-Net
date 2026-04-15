# -*- coding: utf-8 -*-
"""Ultralytics 模型训练入口。"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.external import UltralyticsRunner
from src.utils import get_logger, load_config, send_experiment_notification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ultralytics detectors.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    logger = get_logger("qdcr.train_ultralytics")
    args = parse_args()
    config = load_config(args.config)
    runner = UltralyticsRunner(config)
    start_time = datetime.now()
    try:
        metrics = runner.train()
        end_time = datetime.now()
        send_experiment_notification(config=config, project_root=ROOT, trainer=runner.notify_target(), start_time=start_time, end_time=end_time, success=True, stage="训练", metrics=metrics)
    except Exception as exc:
        end_time = datetime.now()
        logger.exception("[train_ultralytics] training failed")
        send_experiment_notification(config=config, project_root=ROOT, trainer=runner.notify_target(), start_time=start_time, end_time=end_time, success=False, stage="训练", error=exc)
        raise


if __name__ == "__main__":
    main()
