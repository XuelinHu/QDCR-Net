from dataclasses import dataclass


@dataclass
class DetectionLoss:
    """Placeholder detection loss."""

    def __call__(self, predictions, targets):
        return 0.0
