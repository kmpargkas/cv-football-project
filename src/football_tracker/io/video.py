"""Video reading, metadata probing, and letterbox-bar detection/cropping."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from football_tracker.config import IOConfig


@dataclass(frozen=True)
class CropBox:
    """Pixels to crop from each edge of a frame."""

    top: int = 0
    bottom: int = 0
    left: int = 0
    right: int = 0

    @property
    def is_empty(self) -> bool:
        return self.top == self.bottom == self.left == self.right == 0

    def check_fits(self, width: int, height: int) -> None:
        """Raise if this crop would leave nothing of a ``width`` x ``height`` frame."""
        if self.top + self.bottom >= height or self.left + self.right >= width:
            raise ValueError(
                f"io.crop_* ({self.top} top, {self.bottom} bottom, {self.left} left, "
                f"{self.right} right) leaves nothing of a {width}x{height} frame"
            )

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Crop a frame by this box. Returns the frame unchanged if empty."""
        if self.is_empty:
            return frame
        h, w = frame.shape[:2]
        return frame[self.top : h - self.bottom, self.left : w - self.right]


@dataclass(frozen=True)
class VideoMetadata:
    """Basic video stream properties."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


def _open(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    return cap


def _metadata_from(cap: cv2.VideoCapture, path: Path) -> VideoMetadata:
    """Stream properties off an already-open capture, so callers open the file once."""
    return VideoMetadata(
        path=path,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(cap.get(cv2.CAP_PROP_FPS)),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )


def probe_video(path: str | Path) -> VideoMetadata:
    """Read width/height/fps/frame-count from a video file."""
    path = Path(path)
    cap = _open(path)
    try:
        return _metadata_from(cap, path)
    finally:
        cap.release()


def sample_frames(path: str | Path, count: int = 8) -> list[np.ndarray]:
    """Read ``count`` frames evenly spaced across the video."""
    path = Path(path)
    cap = _open(path)
    frames: list[np.ndarray] = []
    try:
        total = max(_metadata_from(cap, path).frame_count, 1)
        for idx in np.linspace(0, total - 1, num=min(count, total), dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
    finally:
        cap.release()
    return frames


def _leading_dark(brightness: np.ndarray, threshold: float, limit: int) -> int:
    """Count contiguous leading entries below ``threshold``, capped at ``limit``."""
    count = 0
    for value in brightness:
        if value < threshold and count < limit:
            count += 1
        else:
            break
    return count


def detect_letterbox(
    frames: list[np.ndarray],
    threshold: float = 20.0,
    max_fraction: float = 0.3,
) -> CropBox:
    """Detect black letterbox/pillarbox bars from sampled frames.

    A row/column counts as a bar only if it stays dark across *all* sampled
    frames (we take the brightest value each ever reaches), so genuinely dark
    play never gets cropped. Cropping is capped at ``max_fraction`` of each
    dimension as a safety limit.
    """
    if not frames:
        return CropBox()
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames])
    row_brightness = gray.mean(axis=2).max(axis=0)  # (H,) brightest each row reaches
    col_brightness = gray.mean(axis=1).max(axis=0)  # (W,) brightest each column reaches
    h, w = gray.shape[1], gray.shape[2]
    return CropBox(
        top=_leading_dark(row_brightness, threshold, int(h * max_fraction)),
        bottom=_leading_dark(row_brightness[::-1], threshold, int(h * max_fraction)),
        left=_leading_dark(col_brightness, threshold, int(w * max_fraction)),
        right=_leading_dark(col_brightness[::-1], threshold, int(w * max_fraction)),
    )


def resolve_crop(io_cfg: IOConfig, video: str | Path) -> CropBox:
    """Auto-detect letterbox bars, or use the manual crop from the config."""
    if io_cfg.auto_letterbox:
        return detect_letterbox(
            sample_frames(video, count=16), threshold=io_cfg.letterbox_threshold
        )
    return CropBox(io_cfg.crop_top, io_cfg.crop_bottom, io_cfg.crop_left, io_cfg.crop_right)


class VideoReader:
    """Iterate frames from a video, with optional letterbox crop and frame range.

    Yields ``(frame_index, frame)`` tuples where ``frame_index`` is the original
    decode index. Usable as a context manager or iterated directly::

        with VideoReader("clip.mp4", crop=box, stride=2) as reader:
            for idx, frame in reader:
                ...
    """

    def __init__(
        self,
        path: str | Path,
        crop: CropBox | None = None,
        start: int = 0,
        stop: int | None = None,
        stride: int = 1,
    ) -> None:
        self.path = Path(path)
        self.crop = crop or CropBox()
        self.start = start
        self.stop = stop
        self.stride = max(1, stride)
        self._cap = _open(self.path)
        self.metadata = _metadata_from(self._cap, self.path)
        self.crop.check_fits(self.metadata.width, self.metadata.height)

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        idx = self.start
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, self.start)
        while True:
            if self.stop is not None and idx >= self.stop:
                break
            ok, frame = self._cap.read()
            if not ok:
                break
            if (idx - self.start) % self.stride == 0:
                yield idx, self.crop.apply(frame)
            idx += 1

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
