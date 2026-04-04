from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import UnderwaterDetectionDataset
from src.utils.config import load_config


def main() -> None:
    config = load_config(ROOT / "configs" / "qdcr_net.yaml")
    dataset_config = config["dataset"]
    dataset = UnderwaterDetectionDataset(
        image_root=ROOT / dataset_config["train_root"],
        annotation_root=(ROOT / dataset_config["train_root"]).parent / "labels",
        enhanced_root=ROOT / dataset_config["enhanced_root"],
        num_classes=dataset_config["num_classes"],
        image_size=dataset_config.get("image_size", 128),
        synthetic_size=dataset_config.get("synthetic_size", 16),
        max_objects=dataset_config.get("max_objects", 8),
    )
    source = dataset[0]["source"] if len(dataset) else "empty"
    print(f"Dataset root: {(ROOT / dataset_config['train_root']).resolve()}")
    print(f"Samples indexed: {len(dataset)}")
    print(f"Source mode: {source}")
    if len(dataset):
        sample = dataset[0]
        print(f"Objects in first sample: {int(sample['gt_classes'].numel())}")
    print("Supported real-data format: image files with YOLO labels stored as matching .txt files.")


if __name__ == "__main__":
    main()
