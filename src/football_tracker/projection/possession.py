"""Team possession from pitch positions: last-touch with acquisition hysteresis.

Acquisition is gated on ball speed because a launched ball projected through the
flat-pitch H sweeps metres from its true position; a fast ball is nobody's. A keeper on
the ball is "keeper": not attributed to either team, and outside the percentages. The
module speaks only "team_a"/"team_b" -- display names live in MatchConfig.
"""

from __future__ import annotations

import numpy as np

from football_tracker.config import PossessionConfig
from football_tracker.projection.artifacts import PossessionState
from football_tracker.projection.to_pitch import BallPitchState, PitchPosition, ball_speeds
from football_tracker.reid.roles import Role
from football_tracker.reid.teamid import TrackLabel


def _team_of(track_id: int, labels: dict[int, TrackLabel]) -> str | None:
    lab = labels.get(track_id)
    if lab is None:
        return None
    if lab.role is Role.TEAM_A:
        return "team_a"
    if lab.role is Role.TEAM_B:
        return "team_b"
    if lab.role is Role.GOALKEEPER:
        # A keeper's kit matches neither team, so the frame is "keeper", not a guess.
        # Still a candidate: otherwise a striker closing down would take the frame.
        return "keeper"
    return None  # referees and unknowns never hold possession


def resolve_possession(
    players: list[PitchPosition],
    ball: list[BallPitchState],
    labels: dict[int, TrackLabel],
    cfg: PossessionConfig,
    fps: float,
) -> list[PossessionState]:
    """One PossessionState per ball frame; last-touch semantics.

    A player acquires by being nearest, within ``radius_m``, while the ball is
    slow, for ``acquire_frames`` consecutive frames. Possession then persists
    through transit and short dropouts; a dropout longer than ``max_unknown_s``
    resets to unknown rather than inventing continuity.
    """
    by_frame: dict[int, list[PitchPosition]] = {}
    for p in players:
        by_frame.setdefault(p.frame_idx, []).append(p)
    speeds = ball_speeds(ball, fps)
    max_unknown = int(round(cfg.max_unknown_s * fps))

    state, holder = "unknown", -1
    cand_tid, cand_team, cand_count = -1, "", 0
    missing_run = 0
    out: list[PossessionState] = []
    for b in sorted(ball, key=lambda s: s.frame_idx):
        if b.xy_m is None or not b.reliable:
            missing_run += 1
            if missing_run > max_unknown:
                state, holder = "unknown", -1
            cand_tid, cand_count = -1, 0
            out.append(PossessionState(b.frame_idx, state, holder))
            continue
        missing_run = 0

        best_tid, best_team, best_dist = -1, "", float("inf")
        for p in by_frame.get(b.frame_idx, []):
            team = _team_of(p.track_id, labels)
            if team is None:
                continue
            d = float(np.linalg.norm(p.xy_m - b.xy_m))
            if d < best_dist:
                best_tid, best_team, best_dist = p.track_id, team, d

        slow = speeds.get(b.frame_idx, float("inf")) <= cfg.max_ball_speed_mps
        if best_dist <= cfg.radius_m and slow and not b.airborne:
            if best_tid == cand_tid:
                cand_count += 1
            else:
                cand_tid, cand_team, cand_count = best_tid, best_team, 1
            if cand_count >= cfg.acquire_frames:
                state, holder = cand_team, cand_tid
        else:
            cand_tid, cand_count = -1, 0
        out.append(PossessionState(b.frame_idx, state, holder))
    return out


def possession_summary(states: list[PossessionState]) -> dict[str, float]:
    """Team shares over attributed frames (sum to 1.0); the other two over all frames."""
    a = sum(s.state == "team_a" for s in states)
    b = sum(s.state == "team_b" for s in states)
    keeper = sum(s.state == "keeper" for s in states)
    attributed = a + b
    return {
        "team_a": a / attributed if attributed else 0.0,
        "team_b": b / attributed if attributed else 0.0,
        "keeper_share": keeper / len(states) if states else 0.0,
        "unknown_share": (len(states) - attributed - keeper) / len(states) if states else 0.0,
    }
