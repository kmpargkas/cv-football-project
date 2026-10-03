"""Pitch calibration: per-frame image↔pitch homography + pitch mask + GMC signal.

An unstable camera has no single homography, so each frame needs its own.
Hand labelled anchors pin H at a few frames and optical flow carries it between them
in both directions. The pass runs once and caches the result.
"""

from football_tracker.homography.anchors import load_anchors
from football_tracker.homography.fuse import fuse_anchors, fused_calibration_stream
from football_tracker.homography.mask import on_pitch
from football_tracker.homography.pitch import PitchSpec
from football_tracker.homography.propagate import FrameMotion, MotionEstimator, propagate_h
from football_tracker.homography.provenance import check_anchor_coverage
from football_tracker.homography.quality import anchor_quality, stream_jitter
from football_tracker.homography.record import CalibrationSource, FrameCalibration
from football_tracker.homography.solve import HomographySolution, project, solve_homography
from football_tracker.homography.store import load_calibration, save_calibration

__all__ = [
    # Calibrating a clip, in the order the stage runs it.
    "load_anchors",
    "check_anchor_coverage",
    "anchor_quality",
    "MotionEstimator",
    "FrameMotion",
    "fuse_anchors",
    "fused_calibration_stream",
    "stream_jitter",
    "save_calibration",
    # Reading a stream back, and what one frame of it holds.
    "load_calibration",
    "FrameCalibration",
    "CalibrationSource",
    # Using a homography.
    "project",
    "on_pitch",
    "propagate_h",
    # Pitch geometry, and fitting one directly.
    "PitchSpec",
    "solve_homography",
    "HomographySolution",
]
