# -*- coding: utf-8 -*-
"""外部检测器适配层。"""

from .faster_rcnn_runner import FasterRCNNRunner
from .ultralytics_runner import UltralyticsRunner

__all__ = ["UltralyticsRunner", "FasterRCNNRunner"]
