from dataclasses import dataclass
from pathlib import Path


@dataclass
class UnderwaterDetectionDataset:
    """Placeholder dataset wrapper for underwater detection experiments."""

    image_root: Path
    annotation_root: Path | None = None
    enhanced_root: Path | None = None

    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int):
        raise IndexError("Dataset placeholder is not implemented yet.")
