from dataclasses import dataclass


@dataclass
class QualityAwareFusion:
    """Placeholder module for quality-aware feature fusion."""

    channels: int

    def __call__(self, raw_feature, enhanced_feature, quality_vector=None):
        return raw_feature
