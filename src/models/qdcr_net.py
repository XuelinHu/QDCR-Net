from dataclasses import dataclass, field

from .modules import CrossResidualBlock, QualityAwareFusion


@dataclass
class QDCRNet:
    """Project-level placeholder for the QDCR-Net detector."""

    num_classes: int
    cross_residual: CrossResidualBlock = field(
        default_factory=lambda: CrossResidualBlock(in_channels=256, out_channels=256)
    )
    fusion: QualityAwareFusion = field(
        default_factory=lambda: QualityAwareFusion(channels=256)
    )

    def forward(self, raw_image, enhanced_image):
        raw_feature = raw_image
        enhanced_feature = enhanced_image
        raw_feature, enhanced_feature = self.cross_residual(raw_feature, enhanced_feature)
        fused_feature = self.fusion(raw_feature, enhanced_feature)
        return {
            "detections": None,
            "fused_feature": fused_feature,
        }
