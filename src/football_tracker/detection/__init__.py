"""Object detection: player / goalkeeper / referee / ball.

YOLO11 behind a model-agnostic detector interface, so the rest of the pipeline
depends on the contract rather than on the model. The ball is the hard sub-target
and gets its own confidence threshold.
"""

from football_tracker.detection.interface import Detections, Detector, ObjectClass
from football_tracker.detection.yolo import YOLODetector

__all__ = [
    "Detections",
    "Detector",
    "ObjectClass",
    "YOLODetector",
]
