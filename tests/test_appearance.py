"""Torso colour descriptor: quality gates and colour fidelity."""

from __future__ import annotations

import numpy as np

from football_tracker.config import TeamIDConfig
from football_tracker.reid.appearance import torso_descriptor

# A box whose torso band lands on rows 70..100 and columns 56..74.
BOX = np.array([50.0, 50.0, 80.0, 150.0])


def _frame(bgr: tuple[int, int, int] = (40, 120, 40), h: int = 200, w: int = 200):
    """A solid-colour BGR frame — green pitch by default."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = bgr
    return frame


def test_red_torso_on_grass_yields_a_red_descriptor():
    frame = _frame()
    frame[70:100, 56:74] = (40, 40, 200)  # BGR red
    sample = torso_descriptor(frame, BOX, TeamIDConfig())
    assert sample is not None
    assert sample[1] > 150.0  # Lab 'a' well above the 128 neutral => red


def test_white_torso_yields_a_high_lightness_descriptor():
    frame = _frame()
    frame[70:100, 56:74] = (235, 235, 235)
    sample = torso_descriptor(frame, BOX, TeamIDConfig())
    assert sample is not None
    assert sample[0] > 220.0
    assert abs(sample[1] - 128.0) < 8.0  # neutral chroma


def test_all_grass_band_is_rejected():
    """Every pixel suppressed as grass => nothing kept => no sample at all."""
    assert torso_descriptor(_frame(), BOX, TeamIDConfig()) is None


def test_box_below_the_height_gate_yields_no_sample():
    frame = _frame()
    frame[50:60, 56:74] = (40, 40, 200)
    short_box = np.array([50.0, 50.0, 80.0, 70.0])  # 20 px tall, gate is 24
    assert torso_descriptor(frame, short_box, TeamIDConfig()) is None


def test_shadowed_pixels_do_not_drag_the_median():
    frame = _frame()
    frame[70:100, 56:74] = (235, 235, 235)  # white kit
    frame[70:100, 56:63] = (5, 5, 5)  # deep cast shadow over part of it
    sample = torso_descriptor(frame, BOX, TeamIDConfig())
    assert sample is not None
    assert sample[0] > 200.0  # shadow suppressed; still white-kit territory


def test_blown_highlight_pixels_are_suppressed():
    frame = _frame()
    frame[70:100, 56:74] = (40, 40, 200)  # red kit
    frame[70:100, 56:63] = (255, 255, 255)  # specular blow-out
    sample = torso_descriptor(frame, BOX, TeamIDConfig())
    assert sample is not None
    assert sample[1] > 150.0  # still reads red, not washed toward neutral


def test_partially_occluded_band_below_the_quality_gate_is_rejected():
    frame = _frame()
    frame[70:100, 70:74] = (40, 40, 200)  # only ~22% of the band is kit
    assert torso_descriptor(frame, BOX, TeamIDConfig()) is None


def test_box_clipped_at_the_frame_edge_does_not_raise():
    frame = _frame()
    frame[70:100, 0:12] = (40, 40, 200)
    sample = torso_descriptor(frame, np.array([-20.0, 50.0, 20.0, 150.0]), TeamIDConfig())
    assert sample is None or sample.shape == (3,)


def test_box_fully_outside_the_frame_yields_no_sample():
    off_frame = np.array([500.0, 500.0, 540.0, 600.0])
    assert torso_descriptor(_frame(), off_frame, TeamIDConfig()) is None


def test_light_blue_and_black_are_distinguishable_by_the_descriptor():
    """A light-blue keeper and a black referee differ in colour, but not by much; the
    goal-zone prior in roles.py is what actually splits them."""
    blue_frame = _frame()
    blue_frame[70:100, 56:74] = (220, 170, 120)  # BGR light blue
    black_frame = _frame()
    black_frame[70:100, 56:74] = (35, 35, 35)
    blue = torso_descriptor(blue_frame, BOX, TeamIDConfig())
    black = torso_descriptor(black_frame, BOX, TeamIDConfig())
    assert blue is not None and black is not None
    assert blue[2] < 128.0  # blue pushes Lab 'b' below neutral
    assert np.linalg.norm(blue - black) > 20.0
