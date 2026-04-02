from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine.trainer import Trainer
from src.utils.config import load_config


def main() -> None:
    config = load_config(ROOT / "configs" / "qdcr_net.yaml")
    trainer = Trainer(config)
    trainer.fit()


if __name__ == "__main__":
    main()
