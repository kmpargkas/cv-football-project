"""Video I/O: letterbox detection and cropping, over synthesized frames."""

from __future__ import annotations

import numpy as np
import pytest

from football_tracker.config import IOConfig
from football_tracker.io import CropBox, detect_letterbox, resolve_crop


def make_frame(height: int = 100, width: int = 120, fill: int = 200) -> np.ndarray:
    """A uniformly bright frame."""
    return np.full((height, width, 3), fill, dtype=np.uint8)


class TestCropBox:
    def test_empty_box_is_a_noop(self):
        frame = make_frame()
        assert CropBox().is_empty
        assert CropBox().apply(frame) is frame

    def test_apply_removes_the_requested_edges(self):
        frame = make_frame(height=100, width=120)
        cropped = CropBox(top=10, bottom=5, left=4, right=6).apply(frame)
        assert cropped.shape[:2] == (100 - 10 - 5, 120 - 4 - 6)

    def test_a_crop_that_consumes_the_frame_is_rejected(self):
        """Silently yielding an empty array corrupts every stage downstream."""
        with pytest.raises(ValueError, match="leaves nothing"):
            CropBox(top=60, bottom=40).check_fits(width=120, height=100)

    def test_a_crop_that_consumes_the_width_is_rejected(self):
        with pytest.raises(ValueError, match="leaves nothing"):
            CropBox(left=70, right=50).check_fits(width=120, height=100)

    def test_a_crop_that_leaves_one_row_is_allowed(self):
        CropBox(top=60, bottom=39).check_fits(width=120, height=100)

    def test_a_detected_letterbox_always_fits(self):
        """detect_letterbox caps at max_fraction, so it cannot produce a rejected box."""
        frame = make_frame(height=100, width=120, fill=0)
        detect_letterbox([frame]).check_fits(width=120, height=100)


class TestResolveCrop:
    """The manual branch needs no video, so it is testable without footage."""

    def test_manual_crop_is_taken_from_the_config(self):
        cfg = IOConfig(auto_letterbox=False, crop_top=7, crop_bottom=3, crop_left=2, crop_right=1)
        assert resolve_crop(cfg, "unused.mp4") == CropBox(top=7, bottom=3, left=2, right=1)

    def test_manual_crop_never_opens_the_video(self):
        cfg = IOConfig(auto_letterbox=False)
        assert resolve_crop(cfg, "does-not-exist.mp4").is_empty


class TestDetectLetterbox:
    def test_no_bars_on_a_bright_frame(self):
        assert detect_letterbox([make_frame()]) == CropBox()

    def test_detects_top_and_bottom_bars(self):
        frame = make_frame()
        frame[:15] = 0  # top bar
        frame[-25:] = 0  # bottom bar
        assert detect_letterbox([frame]) == CropBox(top=15, bottom=25)

    def test_detects_pillarbox_bars(self):
        frame = make_frame()
        frame[:, :8] = 0
        frame[:, -12:] = 0
        assert detect_letterbox([frame]) == CropBox(left=8, right=12)

    def test_dark_but_real_content_is_not_cropped(self):
        """Dark stadium content (~30 brightness) is not a letterbox bar."""
        frame = make_frame()
        frame[:15] = 0  # a genuine black bar
        frame[-20:] = 30  # dark, but real content
        assert detect_letterbox([frame]) == CropBox(top=15, bottom=0)

    def test_a_row_bright_in_any_frame_is_not_a_bar(self):
        """Bars must be dark in *every* sampled frame, not just some."""
        dark = make_frame()
        dark[:15] = 0
        bright = make_frame()
        bright[:15] = 0
        bright[10:15] = 200  # this band lights up in one frame
        assert detect_letterbox([dark, bright]) == CropBox(top=10)

    def test_cropping_is_capped_by_max_fraction(self):
        """An all-black frame must not be cropped away entirely."""
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        box = detect_letterbox([frame], max_fraction=0.3)
        assert box.top == box.bottom == 30
        assert box.left == box.right == 36

    def test_empty_input_returns_empty_box(self):
        assert detect_letterbox([]) == CropBox()
