"""Whether the ball's projected pitch position can be trusted this frame.

Speed stands in for the ``airborne`` flag, which rarely fires: a launched ball
projected through the flat-pitch H sweeps across the pitch at an implausible
ground speed. Uncertainty should be the exception -- a permanently-uncertain
ball means the thresholds are wrong.

Pure function over one frame's state. No model, no video.
"""

from __future__ import annotations

from football_tracker.projection.to_pitch import BallPitchState


def ball_uncertain(
    state: BallPitchState,
    speed_mps: float | None,
    max_plausible_mps: float = 25.0,
) -> tuple[bool, str]:
    """``(is_uncertain, reason)`` for one frame's ball position.

    ``speed_mps`` is ``None`` where no delta exists (clip edges, gaps) and never
    on its own marks a ball uncertain. The speed bound is exclusive.
    """
    reasons: list[str] = []

    if state.xy_m is None:
        reasons.append("no position")
    if state.airborne:
        reasons.append("airborne")
    if state.source == "interpolated":
        reasons.append(state.source)
    if not state.reliable:
        reasons.append("low-confidence homography")
    if speed_mps is not None and speed_mps > max_plausible_mps:
        reasons.append(f"implausible speed {speed_mps:.0f} m/s")

    return bool(reasons), ", ".join(reasons)


def ball_verified(
    state: BallPitchState,
    speed_mps: float | None,
    max_plausible_mps: float = 25.0,
) -> bool:
    """A detected position that passes every check in :func:`ball_uncertain`."""
    if state.source != "detected":
        return False
    uncertain, _ = ball_uncertain(state, speed_mps, max_plausible_mps)
    return not uncertain
