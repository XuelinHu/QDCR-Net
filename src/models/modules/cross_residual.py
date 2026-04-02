from dataclasses import dataclass


@dataclass
class CrossResidualBlock:
    """Placeholder module for bidirectional cross-residual interaction."""

    in_channels: int
    out_channels: int

    def __call__(self, raw_feature, enhanced_feature):
        return raw_feature, enhanced_feature
