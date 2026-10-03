"""Project cached tracks and ball into pitch metres, then attribute possession.

Reads the calibration stream, MOT tracks, team labels and ball trajectory; loads no
model. Writes ``positions.csv``, ``ball_positions.csv`` and ``possession.csv``.

    uv run python scripts/project_tracks.py --config <clip>.yaml \\
        --tracks outputs/<clip>/stitched.txt --teams outputs/<clip>/teams.csv \\
        --ball outputs/<clip>/ball.csv --calibration outputs/<clip>/calibration.npz \\
        --out-dir outputs/<clip>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from football_tracker.config import Config, MatchConfig
from football_tracker.homography.store import load_calibration
from football_tracker.pipeline.cli import (
    check_projection_fps,
    require_contiguous_frames,
    require_file,
)
from football_tracker.projection.artifacts import (
    write_ball_positions_csv,
    write_positions_csv,
    write_possession_csv,
)
from football_tracker.projection.possession import (
    possession_summary,
    resolve_possession,
)
from football_tracker.projection.to_pitch import (
    project_ball_states,
    project_player_tracks,
    smooth_ball,
    smooth_positions,
)
from football_tracker.tracking.artifacts import read_ball_csv, read_mot, read_teams_csv


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Project tracks and ball to pitch metres; attribute possession."
    )
    ap.add_argument("--config", required=True, help="YAML config for the clip.")
    ap.add_argument("--tracks", required=True, help="MOT tracks (stitched.txt or tracks_*.txt)")
    ap.add_argument("--teams", required=True, help="teams.csv from the tracking run")
    ap.add_argument("--ball", required=True, help="ball.csv from the ball pass")
    ap.add_argument("--calibration", default=None, help="default: tracking.calibration_path")
    ap.add_argument(
        "--out-dir",
        required=True,
        help="where positions.csv, ball_positions.csv and possession.csv are written",
    )
    args = ap.parse_args()

    cfg = Config.from_yaml(require_file(args.config, "config"))
    require_contiguous_frames(cfg)
    check_projection_fps(cfg)
    calibration_arg = args.calibration or cfg.tracking.calibration_path
    if calibration_arg is None:
        sys.exit("pass --calibration or set tracking.calibration_path in the config.")
    calibration_path = require_file(calibration_arg, "calibration")
    calibration = load_calibration(calibration_path)
    window = cfg.projection.smooth_window
    out_dir = Path(args.out_dir)

    tracks_path = require_file(args.tracks, "tracks")
    projected_positions = project_player_tracks(read_mot(tracks_path), calibration)
    if not projected_positions:
        sys.exit(
            f"--tracks {tracks_path} produced zero player positions after projection; "
            "the file is likely empty, truncated, or does not match this clip's calibration. "
            "Not writing outputs from it."
        )
    positions = smooth_positions(projected_positions, window)
    write_positions_csv(out_dir / "positions.csv", positions)

    ball_path = require_file(args.ball, "ball")
    projected_ball = project_ball_states(read_ball_csv(ball_path), calibration)
    if not projected_ball:
        sys.exit(
            f"--ball {ball_path} produced zero ball states after projection; "
            "the file is likely empty, truncated, or does not match this clip's calibration. "
            "Not writing outputs from it."
        )
    ball = smooth_ball(projected_ball, window)
    write_ball_positions_csv(out_dir / "ball_positions.csv", ball)
    print(f"wrote {len(positions)} player rows, {len(ball)} ball rows -> {out_dir}")

    labels = read_teams_csv(require_file(args.teams, "teams"))
    states = resolve_possession(
        positions, ball, labels, cfg.projection.possession, cfg.projection.fps
    )
    write_possession_csv(out_dir / "possession.csv", states)
    s = possession_summary(states)
    names = cfg.match or MatchConfig()
    print(
        f"possession: {names.team_a_name} {s['team_a']:.1%} - "
        f"{names.team_b_name} {s['team_b']:.1%} "
        f"(keeper {s['keeper_share']:.1%}, unknown {s['unknown_share']:.1%} "
        f"of {len(states)} frames)"
    )


if __name__ == "__main__":
    main()
