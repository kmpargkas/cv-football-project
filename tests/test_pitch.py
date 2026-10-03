"""A non-standard pitch rescales the keypoints rather than raising."""

from __future__ import annotations

import numpy as np

from football_tracker.homography.pitch import PitchSpec


def test_keypoints_scale_with_the_configured_pitch():
    standard = PitchSpec()
    small = PitchSpec(length=100.0, width=64.0)
    assert standard.keypoints.shape == small.keypoints.shape
    assert not np.allclose(standard.keypoints, small.keypoints)
    assert small.keypoints[:, 0].max() == 100.0
    assert small.keypoints[:, 1].max() == 64.0
