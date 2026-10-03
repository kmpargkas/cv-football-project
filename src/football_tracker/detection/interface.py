"""The detector contract: a canonical class taxonomy, a typed result, and a Protocol.

This module defines *what a detector is* independent of which model implements it,
so the pipeline depends on the contract rather than on a model. It imports
nothing heavy - every model maps its native output into the types defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol, runtime_checkable

import numpy as np


class ObjectClass(IntEnum):
    """The four detection classes, with the canonical IDs the whole codebase uses."""

    BALL = 0
    GOALKEEPER = 1
    PLAYER = 2
    REFEREE = 3


@dataclass(frozen=True, eq=False)
class Detections:
    """One frame's detections: parallel arrays of boxes, scores, and class IDs.

    Mirrors the flat-array shape of ``supervision.Detections`` but stays a thin,
    import-light project type. ``xyxy`` is ``(N, 4)`` float pixels ``[x1, y1, x2, y2]``;
    ``confidence`` and ``class_id`` are ``(N,)``. ``class_id`` values are
    :class:`ObjectClass` integers.
    """

    xyxy: np.ndarray
    confidence: np.ndarray
    class_id: np.ndarray

    def __post_init__(self) -> None:
        xyxy = np.asarray(self.xyxy, dtype=np.float32).reshape(-1, 4)
        confidence = np.asarray(self.confidence, dtype=np.float32).reshape(-1)
        class_id = np.asarray(self.class_id, dtype=np.int64).reshape(-1)
        if not (len(xyxy) == len(confidence) == len(class_id)):
            raise ValueError(
                f"Mismatched lengths: {len(xyxy)} boxes, {len(confidence)} scores, "
                f"{len(class_id)} class IDs."
            )
        # Frozen dataclass: assign the normalized arrays through object.__setattr__.
        object.__setattr__(self, "xyxy", xyxy)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "class_id", class_id)

    def __len__(self) -> int:
        return len(self.xyxy)

    @classmethod
    def empty(cls) -> Detections:
        """An empty set of detections (no boxes)."""
        return cls(
            xyxy=np.zeros((0, 4), dtype=np.float32),
            confidence=np.zeros((0,), dtype=np.float32),
            class_id=np.zeros((0,), dtype=np.int64),
        )

    @property
    def foot_points(self) -> np.ndarray:
        """``(N, 2)`` bottom-center of each box.

        Where a player touches the pitch, so it is the point that projects through
        the homography - not the box center.
        """
        x = (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2.0
        y = self.xyxy[:, 3]
        return np.stack([x, y], axis=1)

    def filter_by_class(self, *classes: ObjectClass) -> Detections:
        """Keep only detections whose class is one of ``classes``."""
        wanted = {int(c) for c in classes}
        mask = np.array([int(c) in wanted for c in self.class_id], dtype=bool)
        return self._select(mask)

    def with_min_confidence(self, threshold: float) -> Detections:
        """Keep only detections scoring at least ``threshold``."""
        return self._select(self.confidence >= threshold)

    def _select(self, mask: np.ndarray) -> Detections:
        return Detections(
            xyxy=self.xyxy[mask],
            confidence=self.confidence[mask],
            class_id=self.class_id[mask],
        )


@runtime_checkable
class Detector(Protocol):
    """Anything that turns a frame into :class:`Detections`.

    The pipeline depends only on this Protocol, so a detector implementation can be
    swapped without touching it.
    """

    def detect(self, frame: np.ndarray) -> Detections:
        """Detect objects in a single BGR frame (already letterbox-cropped upstream)."""
        ...
