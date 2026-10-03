"""Per-anchor quality reporting in the calibrate stage.

It lives in ``main()``, so these run it end to end over a six-frame synthetic clip.
"""

from __future__ import annotations

import json
import sys

import cv2
import numpy as np
import pytest

from football_tracker.homography.pitch import PitchSpec

SPEC = PitchSpec()
FRAME_H, FRAME_W = 360, 640
IMAGE_CORNERS = np.array([(40, 30), (600, 40), (620, 330), (20, 320)], dtype=np.float32)
# Well spread: both corners at each end, both halfway points, a goal-box and a penalty-box
# corner -- enough for a solve that reaches both ends of the pitch.
VERTICES = (0, 5, 24, 29, 13, 16, 8, 21)


def _write_clip(path, frames=6):
    """A textured clip -- MotionEstimator needs corners to track between frames."""
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur(rng.integers(0, 255, (FRAME_H, FRAME_W, 3), dtype=np.uint8), (5, 5), 0)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (FRAME_W, FRAME_H))
    for i in range(frames):
        writer.write(np.roll(base, i * 2, axis=1))
    writer.release()
    if not path.exists() or path.stat().st_size < 1000:
        pytest.skip("cv2.VideoWriter cannot encode mp4v here")
    return path


def _clicks(bad=()):
    """Exact clicks from a known homography; ``bad`` indices are displaced off-feature."""
    H = cv2.getPerspectiveTransform(SPEC.boundary.astype(np.float32), IMAGE_CORNERS)
    points = []
    for k, vertex in enumerate(VERTICES):
        pitch = SPEC.keypoints[vertex]
        q = H @ np.array([pitch[0], pitch[1], 1.0])
        xy = q[:2] / q[2] + (np.array([25.0, -25.0]) if k in bad else 0.0)
        points.append(
            {
                "image": [float(xy[0]), float(xy[1])],
                "pitch": [float(pitch[0]), float(pitch[1])],
                "vertex": int(vertex) + 1,
            }
        )
    return points


def _run(tmp_path, monkeypatch, frames, *extra):
    from scripts.calibrate_video import main

    clip = _write_clip(tmp_path / "clip.mp4")
    anchors = tmp_path / "pitch_points.json"
    anchors.write_text(json.dumps({"frames": frames}))
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibrate_video.py",
            "--video",
            str(clip),
            "--anchors",
            str(anchors),
            "--output-dir",
            str(out_dir),
            *extra,
        ],
    )
    main()
    return out_dir


CLEAN = {"0": _clicks(), "4": _clicks()}
ONE_POOR = {"0": _clicks(), "4": _clicks(bad=(0, 1, 2))}


def test_a_poor_anchor_is_named_with_its_worst_clicks(tmp_path, monkeypatch, capsys):
    _run(tmp_path, monkeypatch, ONE_POOR)
    err = capsys.readouterr().err
    assert "1 of 2 anchors" in err
    assert "frame 4" in err
    assert "revisit points" in err


def test_a_clean_anchor_set_says_nothing(tmp_path, monkeypatch, capsys):
    """Silent when there is nothing to report, like the coverage check above it."""
    _run(tmp_path, monkeypatch, CLEAN)
    assert "anchors are below" not in capsys.readouterr().err


def test_a_poor_anchor_does_not_stop_the_run(tmp_path, monkeypatch):
    """Reported only -- the held-out gate stays the sole arbiter."""
    out_dir = _run(tmp_path, monkeypatch, ONE_POOR)
    assert (out_dir / "calibration.npz").exists()


def test_every_anchor_is_recorded_in_the_metrics(tmp_path, monkeypatch):
    """The aggregate medians cannot say which anchor is bad; this can."""
    out_dir = _run(tmp_path, monkeypatch, ONE_POOR)
    quality = json.loads((out_dir / "calibration_metrics.json").read_text())["anchor_quality"]
    assert set(quality) == {"0", "4"}
    assert quality["0"]["inlier_ratio"] == 1.0
    assert quality["4"]["inlier_ratio"] < 0.7
    assert quality["4"]["outlier_indices"]
    assert quality["0"]["end_reach"] == "BOTH"


def _report_lines(captured: str) -> list[str]:
    """Only the per-anchor table indents a line with ``frame``."""
    return [line for line in captured.splitlines() if line.startswith("  frame ")]


def test_anchor_report_lists_every_anchor_not_only_the_poor_ones(tmp_path, monkeypatch, capsys):
    _run(tmp_path, monkeypatch, CLEAN, "--anchor-report")
    assert len(_report_lines(capsys.readouterr().out)) == 2


def test_without_the_flag_a_clean_run_lists_no_anchors_at_all(tmp_path, monkeypatch, capsys):
    """The warning is unconditional; the full table is what the flag buys."""
    _run(tmp_path, monkeypatch, CLEAN)
    assert _report_lines(capsys.readouterr().out) == []
