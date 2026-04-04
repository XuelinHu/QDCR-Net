import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine.trainer import Trainer
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a detection checkpoint and report mAP metrics.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "qdcr_net.yaml",
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    trainer = Trainer(config)
    metrics = trainer.evaluate()
    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    main()
