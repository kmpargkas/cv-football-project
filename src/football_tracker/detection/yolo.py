"""Ultralytics YOLO11 detector — the implementation of the Detector protocol.

The adapter between a numpy frame and our typed :class:`Detections`: it runs YOLO11,
remaps the model's native class IDs onto :class:`ObjectClass`, and applies the
configured thresholds. ``ultralytics`` is imported lazily so importing the package
stays light.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from football_tracker.detection.interface import Detections, ObjectClass

if TYPE_CHECKING:
    from football_tracker.config import DetectionConfig

# Class-name -> canonical class. Keyed by name (not native ID) so it is robust to
# whatever order a given model happens to define its classes in.
_NAME_TO_CLASS: dict[str, ObjectClass] = {
    "ball": ObjectClass.BALL,
    "goalkeeper": ObjectClass.GOALKEEPER,
    "player": ObjectClass.PLAYER,
    "referee": ObjectClass.REFEREE,
}


def build_id_map(names: dict[int, str]) -> dict[int, ObjectClass]:
    """Map a model's native class IDs onto :class:`ObjectClass`.

    ``names`` is Ultralytics' ``{id: name}`` dict. Names we don't model (e.g. "crowd")
    are dropped, so their detections never enter the pipeline.
    """
    id_map: dict[int, ObjectClass] = {}
    for native_id, name in names.items():
        canonical = _NAME_TO_CLASS.get(name.strip().lower())
        if canonical is not None:
            id_map[int(native_id)] = canonical
    return id_map


def translate(
    xyxy: np.ndarray,
    conf: np.ndarray,
    cls: np.ndarray,
    id_map: dict[int, ObjectClass],
    ball_conf: float | None = None,
) -> Detections:
    """Turn raw model arrays into :class:`Detections`, remapping and filtering classes.

    Detections whose native class is not in ``id_map`` are dropped. When ``ball_conf``
    is set, ball detections below it are rejected (the ball needs a stricter bar than
    players — it is small and easily confused with white distractors).
    """
    xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
    conf = np.asarray(conf, dtype=np.float32).reshape(-1)
    cls = np.asarray(cls).reshape(-1).astype(int)
    if not (len(xyxy) == len(conf) == len(cls)):
        raise ValueError(
            f"Mismatched lengths: {len(xyxy)} boxes, {len(conf)} scores, {len(cls)} classes."
        )

    keep_boxes: list[np.ndarray] = []
    keep_conf: list[float] = []
    keep_cls: list[int] = []
    for i, native in enumerate(cls):
        canonical = id_map.get(int(native))
        if canonical is None:
            continue
        if ball_conf is not None and canonical == ObjectClass.BALL and conf[i] < ball_conf:
            continue
        keep_boxes.append(xyxy[i])
        keep_conf.append(float(conf[i]))
        keep_cls.append(int(canonical))

    if not keep_boxes:
        return Detections.empty()
    return Detections(
        xyxy=np.array(keep_boxes, dtype=np.float32),
        confidence=np.array(keep_conf, dtype=np.float32),
        class_id=np.array(keep_cls, dtype=np.int64),
    )


def _to_numpy(x: Any) -> np.ndarray:
    """Best-effort conversion of an Ultralytics tensor (or array) to numpy."""
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x)


class YOLODetector:
    """A :class:`~football_tracker.detection.interface.Detector` backed by YOLO11."""

    def __init__(
        self,
        model: Any,
        *,
        imgsz: int = 1280,
        conf: float = 0.25,
        iou: float = 0.7,
        max_det: int = 300,
        ball_conf: float | None = None,
        agnostic_nms: bool = True,
        device: str = "cpu",
    ) -> None:
        self._model = model
        self._imgsz = imgsz
        self._conf = conf
        self._iou = iou
        self._max_det = max_det
        self._agnostic_nms = agnostic_nms
        self._ball_conf = ball_conf
        self._device = device
        self._id_map = build_id_map(model.names)

    @classmethod
    def from_config(cls, config: DetectionConfig, device: str = "auto") -> YOLODetector:
        """Load YOLO11 weights from ``config`` and resolve the device."""
        from ultralytics import YOLO

        from football_tracker.utils import resolve_device

        if config.weights is None:
            raise ValueError("DetectionConfig.weights is not set — no model to load.")
        weights = Path(config.weights)
        if not weights.exists():
            raise FileNotFoundError(
                f"Weights not found: {weights}. Pull them with scripts/pull_weights.py."
            )
        model = YOLO(str(weights))
        return cls(
            model,
            imgsz=config.imgsz,
            conf=config.conf,
            iou=config.iou,
            max_det=config.max_det,
            ball_conf=config.ball_conf,
            agnostic_nms=config.agnostic_nms,
            device=resolve_device(device),
        )

    def reconfigured(
        self,
        *,
        imgsz: int | None = None,
        conf: float | None = None,
        ball_conf: float | None | Any = ...,
    ) -> YOLODetector:
        """A new wrapper sharing this loaded model, with overridden thresholds.

        Used by the ball ROI pass: same weights, but native-resolution input and a
        permissive confidence floor. ``imgsz``/``conf`` keep the current value when
        ``None``; ``ball_conf`` keeps it when omitted (``None`` is a meaningful
        value - "no stricter ball threshold").
        """
        return YOLODetector(
            self._model,
            imgsz=self._imgsz if imgsz is None else imgsz,
            conf=self._conf if conf is None else conf,
            iou=self._iou,
            max_det=self._max_det,
            ball_conf=self._ball_conf if ball_conf is ... else ball_conf,
            agnostic_nms=self._agnostic_nms,
            device=self._device,
        )

    def detect(self, frame: np.ndarray) -> Detections:
        """Detect objects in a single BGR frame."""
        results = self._model.predict(
            frame,
            imgsz=self._imgsz,
            conf=self._conf,
            iou=self._iou,
            max_det=self._max_det,
            agnostic_nms=self._agnostic_nms,
            device=self._device,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return Detections.empty()
        return translate(
            xyxy=_to_numpy(boxes.xyxy),
            conf=_to_numpy(boxes.conf),
            cls=_to_numpy(boxes.cls),
            id_map=self._id_map,
            ball_conf=self._ball_conf,
        )
