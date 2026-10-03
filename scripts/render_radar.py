"""Render the 2D radar MP4: players and ball as dots on a drawn pitch.

Dots take the measured kit colours from ``palette.json`` -- both teams, each
goalkeeper's own kit, and the referee -- with placeholder red/white and black
when it is absent (keepers then grey, as unknown-role tracks always are).
The ball is a small yellow dot, hidden on missing frames; where its position
cannot be trusted it becomes a hollow ring and the banner says why. The banner
leads with the frame index and, given ``possession.csv``, the running split.

    uv run python scripts/render_radar.py --config <clip>.yaml \\
        --positions outputs/<clip>/positions.csv --ball outputs/<clip>/ball_positions.csv \\
        --teams outputs/<clip>/teams.csv --palette outputs/<clip>/palette.json \\
        --possession outputs/<clip>/possession.csv --out outputs/<clip>/radar.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from football_tracker.config import Config
from football_tracker.pipeline.cli import (
    check_projection_fps,
    require_contiguous_frames,
    require_file,
)
from football_tracker.projection.artifacts import (
    PossessionState,
    read_ball_positions_csv,
    read_positions_csv,
    read_possession_csv,
)
from football_tracker.projection.radar import Dot, draw_radar_frame, hex_to_bgr, render_pitch
from football_tracker.projection.to_pitch import PitchPosition, ball_speeds
from football_tracker.projection.uncertainty import ball_uncertain
from football_tracker.reid.teamid import KitColours
from football_tracker.tracking.artifacts import read_palette, read_teams_csv

REFEREE_BGR = (30, 30, 30)
BALL_BGR = (0, 215, 255)
DEFAULT_A, DEFAULT_B = "#d62828", "#e8e8e8"


def main() -> None:
    ap = argparse.ArgumentParser(description="Render the radar MP4 from projected positions.")
    ap.add_argument("--config", required=True, help="YAML config for the clip.")
    ap.add_argument("--positions", required=True, help="positions.csv from project_tracks.py")
    ap.add_argument("--ball", required=True, help="ball_positions.csv from project_tracks.py")
    ap.add_argument("--possession", default=None, help="possession.csv; adds the running split")
    ap.add_argument("--teams", required=True, help="teams.csv from the tracking run")
    ap.add_argument(
        "--palette",
        default=None,
        help="palette.json from the tracking run; dots take its measured kit colours",
    )
    ap.add_argument("--out", required=True, help="output radar MP4")
    args = ap.parse_args()

    cfg = Config.from_yaml(require_file(args.config, "config"))
    require_contiguous_frames(cfg)
    check_projection_fps(cfg)
    spec = cfg.pitch.to_spec()
    scale, margin = cfg.projection.scale_px_per_m, cfg.projection.margin_px

    positions = read_positions_csv(require_file(args.positions, "positions"))
    if not positions:
        sys.exit(f"--positions {args.positions} contains zero rows; nothing to render.")
    ball_rows = read_ball_positions_csv(require_file(args.ball, "ball"))
    if not ball_rows:
        sys.exit(f"--ball {args.ball} contains zero rows; nothing to render.")
    ball = {s.frame_idx: s for s in ball_rows}
    # Speed is a delta over consecutive frames, so it is computed once over the clip.
    speeds = ball_speeds(ball_rows, cfg.projection.fps)
    labels = read_teams_csv(require_file(args.teams, "teams"))
    possession: dict[int, PossessionState] = {}
    if args.possession:
        possession_rows = read_possession_csv(require_file(args.possession, "possession"))
        if not possession_rows:
            sys.exit(f"--possession {args.possession} contains zero rows; nothing to render.")
        possession = {s.frame_idx: s for s in possession_rows}

    name_a = cfg.match.team_a_name if cfg.match else "Team A"
    name_b = cfg.match.team_b_name if cfg.match else "Team B"
    if args.palette and Path(args.palette).exists():
        colours = KitColours.from_palette(read_palette(args.palette))
    else:
        print(
            "warning: no palette.json; radar dots use placeholder colours, not the measured kits",
            file=sys.stderr,
        )
        colours = KitColours(
            team=(hex_to_bgr(DEFAULT_A), hex_to_bgr(DEFAULT_B)),
            goalkeepers=(),
            referee=REFEREE_BGR,
        )

    by_frame: dict[int, list[PitchPosition]] = {}
    for p in positions:
        by_frame.setdefault(p.frame_idx, []).append(p)
    frames = sorted(set(by_frame) | set(ball))

    background = render_pitch(spec, scale, margin)
    h, w = background.shape[:2]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), cfg.projection.fps, (w, h)
    )
    if not writer.isOpened():
        sys.exit(
            f"cv2.VideoWriter failed to open {out_path}; check the mp4v codec is available "
            "and the directory is writable."
        )
    counts = {"team_a": 0, "team_b": 0}
    for f in frames:
        dots = []
        for p in by_frame.get(f, []):
            color = colours.for_label(labels.get(p.track_id))
            dots.append(Dot(p.xy_m, color, radius_px=6, outline=False))
        b = ball.get(f)
        uncertain, reason = False, ""
        if b is not None:
            uncertain, reason = ball_uncertain(b, speeds.get(f), cfg.ball.airborne_speed_ms)
        if b is not None and b.xy_m is not None:
            # Uncertain: a hollow ring at 2x radius, no outline, so it reads as one shape.
            dots.append(
                Dot(
                    b.xy_m,
                    BALL_BGR,
                    radius_px=8 if uncertain else 4,
                    outline=not uncertain,
                    filled=not uncertain,
                )
            )

        banner = f"frame {f}"
        state = possession.get(f)
        if state is not None:
            if state.state in counts:
                counts[state.state] += 1
            total = counts["team_a"] + counts["team_b"]
            if total:
                banner += (
                    f"  |  {name_a} {counts['team_a'] / total:.0%} - "
                    f"{name_b} {counts['team_b'] / total:.0%}"
                )
            holder = {"team_a": name_a, "team_b": name_b, "keeper": "keeper"}.get(state.state)
            banner += f"  |  ball: {holder}" if holder else "  |  ball: -"
        if uncertain:
            banner += f"  |  ball uncertain: {reason}"
        writer.write(draw_radar_frame(background, dots, banner, scale, margin))
    writer.release()
    print(f"wrote {len(frames)} radar frames -> {out_path}")


if __name__ == "__main__":
    main()
