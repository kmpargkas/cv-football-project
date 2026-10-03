"""Player tracking and the separate ball tracker.

Players: BoT-SORT over the detector's boxes, with the calibration stage's camera
motion injected and off-pitch detections rejected. Ball: per-frame candidates from a
Kalman-placed ROI re-detection, then an offline min-cost path with interpolation and
airborne flags. Both stages write and read their outputs through ``artifacts``.
"""

from football_tracker.tracking.artifacts import (
    mot_row,
    read_ball_csv,
    read_candidates_csv,
    read_mot,
    read_palette,
    read_teams_csv,
    write_ball_csv,
    write_candidates_csv,
    write_palette,
    write_teams_csv,
)
from football_tracker.tracking.ball import BallCandidate, BallObserver
from football_tracker.tracking.ball_trajectory import BallState, build_trajectory
from football_tracker.tracking.tracker import PlayerTracker, TrackedObjects

__all__ = [
    # Tracking players.
    "PlayerTracker",
    "TrackedObjects",
    # Tracking the ball: per-frame candidates, then the offline trajectory.
    "BallObserver",
    "BallCandidate",
    "build_trajectory",
    "BallState",
    # The stage outputs on disk.
    "mot_row",
    "read_mot",
    "write_teams_csv",
    "read_teams_csv",
    "write_palette",
    "read_palette",
    "write_ball_csv",
    "read_ball_csv",
    "write_candidates_csv",
    "read_candidates_csv",
]
