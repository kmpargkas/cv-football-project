"""Video I/O: reading, metadata probing, and letterbox cropping."""

from football_tracker.io.video import (
    CropBox,
    VideoMetadata,
    VideoReader,
    detect_letterbox,
    probe_video,
    resolve_crop,
    sample_frames,
)

__all__ = [
    "CropBox",
    "VideoMetadata",
    "VideoReader",
    "detect_letterbox",
    "probe_video",
    "resolve_crop",
    "sample_frames",
]
