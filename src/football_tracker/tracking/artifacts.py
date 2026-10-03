"""On-disk artifacts that cross pipeline stages: tracks, teams, palette, ball.

Readers and writers live together so the two sides of each format cannot drift
apart — the ±1 MOT shift in particular is silent when it goes wrong.

Every ``frame`` field is the 0-based decode index yielded by
:class:`~football_tracker.io.video.VideoReader`, so rows from different stages join
on frame number as long as the stages shared an I/O config.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from football_tracker.reid.roles import Role
from football_tracker.reid.teamid import TeamPalette, TrackLabel
from football_tracker.tracking.ball import BallCandidate
from football_tracker.tracking.ball_trajectory import BallState
from football_tracker.tracking.tracker import TrackedObjects

BALL_CSV_HEADER = "frame,x,y,source,airborne,confidence"
CANDIDATES_CSV_HEADER = "frame,x,y,confidence,source"
TEAMS_CSV_HEADER = "track_id,team,role,confidence,n_samples,n_rejected"


# --- ball trajectory ---------------------------------------------------------------


def write_ball_csv(path: str | Path, states: list[BallState]) -> None:
    """Ball trajectory CSV, one row per frame; ``missing`` frames keep their row with empty x/y."""
    lines = [BALL_CSV_HEADER]
    for s in states:
        x = f"{s.xy[0]:.2f}" if s.xy is not None else ""
        y = f"{s.xy[1]:.2f}" if s.xy is not None else ""
        lines.append(f"{s.frame_idx},{x},{y},{s.source},{int(s.airborne)},{s.confidence:.3f}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def read_ball_csv(path: str | Path) -> dict[int, BallState]:
    """Inverse of :func:`write_ball_csv`, keyed by frame index."""
    out: dict[int, BallState] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("frame"):
            continue
        frame, x, y, source, airborne, confidence = line.split(",")
        xy = None if x == "" or y == "" else np.array([float(x), float(y)], dtype=np.float64)
        out[int(frame)] = BallState(
            frame_idx=int(frame),
            xy=xy,
            source=source,
            airborne=bool(int(airborne)),
            confidence=float(confidence),
        )
    return out


def write_candidates_csv(path: str | Path, candidates: list[BallCandidate]) -> None:
    """Raw per-frame ball candidates, before the path solve picked among them."""
    lines = [CANDIDATES_CSV_HEADER]
    lines += [
        f"{c.frame_idx},{c.xy[0]:.2f},{c.xy[1]:.2f},{c.confidence:.3f},{c.source}"
        for c in candidates
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def read_candidates_csv(path: str | Path) -> dict[int, list[tuple[float, float, float, str]]]:
    """Raw ball candidates as ``{frame: [(x, y, confidence, source), ...]}``.

    Used by the renderer's ``--candidates`` overlay to show what the path solver
    had available versus what it chose.
    """
    out: dict[int, list[tuple[float, float, float, str]]] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("frame"):
            continue
        frame, x, y, confidence, source = line.split(",")
        out.setdefault(int(frame), []).append((float(x), float(y), float(confidence), source))
    return out


# --- team / role labels --------------------------------------------------


def write_teams_csv(path: str | Path, labels: dict[int, TrackLabel]) -> None:
    """One row per track: its team, role, and how much evidence stood behind them.

    Keyed on ``track_id`` rather than frame because a role is decided once over a track's
    whole life. ``confidence`` is the winning vote share; a long track with a low share
    usually contains an ID switch between two players on opposite teams.
    """
    rows = [TEAMS_CSV_HEADER]
    for track_id in sorted(labels):
        label = labels[track_id]
        rows.append(
            f"{track_id},{label.team},{label.role.value},{label.confidence:.4f},"
            f"{label.n_samples},{label.n_rejected}"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n")


def read_teams_csv(path: str | Path) -> dict[int, TrackLabel]:
    """Inverse of :func:`write_teams_csv`, keyed by track id."""
    out: dict[int, TrackLabel] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("track_id"):
            continue
        track_id, team, role, confidence, n_samples, n_rejected = line.split(",")
        out[int(track_id)] = TrackLabel(
            role=Role(role),
            team=int(team),
            confidence=float(confidence),
            n_samples=int(n_samples),
            n_rejected=int(n_rejected),
        )
    return out


def write_palette(path: str | Path, palette: TeamPalette) -> None:
    """The frozen colour model, plus the provenance of the fit that produced it, so a
    wrong palette - which still assigns confidently - can be diagnosed after the fact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(palette.to_dict(), indent=2, sort_keys=True) + "\n")


def read_palette(path: str | Path) -> TeamPalette:
    """Inverse of :func:`write_palette`."""
    return TeamPalette.from_dict(json.loads(Path(path).read_text()))


# --- player tracks (MOT-Challenge) --------------------------------------------------


def mot_row(frame_idx: int, track_id: int, class_id: int, xyxy: np.ndarray) -> str:
    """One MOT-Challenge hypothesis row: ``frame,id,x,y,w,h,conf,class,vis``.

    X/Y are the box top-left **plus 1**, as the MOT-Challenge format is 1-indexed.
    W/H are unshifted. The frame field stays the raw 0-based decode index —
    :func:`read_mot` undoes exactly this and nothing else.
    """
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    x, y, w, h = x1 + 1, y1 + 1, x2 - x1, y2 - y1
    return f"{frame_idx},{int(track_id)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,{int(class_id)},-1"


def read_mot(path: str | Path) -> dict[int, TrackedObjects]:
    """Inverse of :func:`mot_row`, grouped into one :class:`TrackedObjects` per frame."""
    rows: dict[int, list[tuple[int, int, list[float]]]] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        frame, tid, x, y, w, h = (float(p) for p in parts[:6])
        cls = int(float(parts[7]))
        # Undo the +1 written by mot_row; W/H were never shifted.
        x1, y1 = x - 1.0, y - 1.0
        rows.setdefault(int(frame), []).append((int(tid), cls, [x1, y1, x1 + w, y1 + h]))

    out: dict[int, TrackedObjects] = {}
    for frame, entries in rows.items():
        out[frame] = TrackedObjects(
            xyxy=np.array([e[2] for e in entries], dtype=np.float32).reshape(-1, 4),
            id=np.array([e[0] for e in entries], dtype=np.int64),
            confidence=np.ones(len(entries), dtype=np.float32),
            class_id=np.array([e[1] for e in entries], dtype=np.int64),
        )
    return out
