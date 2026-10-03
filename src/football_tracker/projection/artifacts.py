"""CSV artifacts of the projection pass; every ``frame`` is the 0-based decode index."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from football_tracker.projection.to_pitch import BallPitchState, PitchPosition

POSITIONS_CSV_HEADER = "frame,track_id,x_m,y_m,reliable"
BALL_POSITIONS_CSV_HEADER = "frame,x_m,y_m,source,airborne,reliable"
POSSESSION_CSV_HEADER = "frame,state,track_id"


@dataclass(frozen=True)
class PossessionState:
    """Which team holds possession on one frame."""

    frame_idx: int
    state: str  # "team_a" | "team_b" | "keeper" | "unknown"
    track_id: int  # who last acquired the ball, not who is nearest; -1 when unknown


def write_positions_csv(path: str | Path, positions: list[PitchPosition]) -> None:
    lines = [POSITIONS_CSV_HEADER]
    lines += [
        f"{p.frame_idx},{p.track_id},{p.xy_m[0]:.3f},{p.xy_m[1]:.3f},{int(p.reliable)}"
        for p in positions
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def read_positions_csv(path: str | Path) -> list[PitchPosition]:
    out: list[PitchPosition] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("frame"):
            continue
        frame, tid, x, y, reliable = line.split(",")
        out.append(
            PitchPosition(
                frame_idx=int(frame),
                track_id=int(tid),
                xy_m=np.array([float(x), float(y)], dtype=np.float64),
                reliable=bool(int(reliable)),
            )
        )
    return out


def write_ball_positions_csv(path: str | Path, ball: list[BallPitchState]) -> None:
    """Missing positions are written with empty x/y fields, not omitted."""
    lines = [BALL_POSITIONS_CSV_HEADER]
    for s in ball:
        x = f"{s.xy_m[0]:.3f}" if s.xy_m is not None else ""
        y = f"{s.xy_m[1]:.3f}" if s.xy_m is not None else ""
        lines.append(f"{s.frame_idx},{x},{y},{s.source},{int(s.airborne)},{int(s.reliable)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def read_ball_positions_csv(path: str | Path) -> list[BallPitchState]:
    out: list[BallPitchState] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("frame"):
            continue
        frame, x, y, source, airborne, reliable = line.split(",")
        xy = None if x == "" else np.array([float(x), float(y)], dtype=np.float64)
        out.append(
            BallPitchState(
                frame_idx=int(frame),
                xy_m=xy,
                source=source,
                airborne=bool(int(airborne)),
                reliable=bool(int(reliable)),
            )
        )
    return out


def write_possession_csv(path: str | Path, states: list[PossessionState]) -> None:
    lines = [POSSESSION_CSV_HEADER]
    lines += [f"{s.frame_idx},{s.state},{s.track_id}" for s in states]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def read_possession_csv(path: str | Path) -> list[PossessionState]:
    out: list[PossessionState] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("frame"):
            continue
        frame, state, tid = line.split(",")
        out.append(PossessionState(frame_idx=int(frame), state=state, track_id=int(tid)))
    return out
