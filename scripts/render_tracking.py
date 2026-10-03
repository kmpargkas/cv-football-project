"""Draw the cached player tracks and ball path onto the footage.

With ``--palette``, ``--positions`` and ``--ball-positions`` this writes the
deliverable: an outline ellipse at each player's feet in the measured kit colour, a
black ring round detected balls, and a glass minimap of the projected positions at
the bottom of the frame, and a number under each ellipse in order of appearance
(``--no-ids`` drops it) -- no boxes, no other text. Without them it writes the
debug render: boxes and ``#<id>`` labels, the ball's fading trail by source, and
optionally every raw ball candidate.

Every input must come from the same I/O config (crop, frame range, stride): rows
join on the decode frame index, so a different crop silently misaligns the overlays.

    uv run python scripts/render_tracking.py --config <clip>.yaml \\
        --tracks outputs/<clip>/stitched.txt --ball outputs/<clip>/ball.csv \\
        --teams outputs/<clip>/teams.csv --palette outputs/<clip>/palette.json \\
        --positions outputs/<clip>/positions.csv \\
        --ball-positions outputs/<clip>/ball_positions.csv \\
        --output outputs/<clip>/demo.mp4
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from football_tracker.config import Config
from football_tracker.io.video import VideoReader, resolve_crop
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
from football_tracker.projection.radar import (
    Dot,
    Minimap,
    draw_minimap,
    draw_possession_bar,
    minimap_geometry,
)
from football_tracker.projection.to_pitch import BallPitchState, PitchPosition, ball_speeds
from football_tracker.projection.uncertainty import ball_verified
from football_tracker.reid.teamid import KitColours, TrackLabel
from football_tracker.tracking.annotate import (
    BallAnnotator,
    KitAnnotator,
    TrackAnnotator,
    draw_ball_marker,
)
from football_tracker.tracking.artifacts import (
    read_ball_csv,
    read_candidates_csv,
    read_mot,
    read_palette,
    read_teams_csv,
)
from football_tracker.tracking.ball_trajectory import BallState
from football_tracker.tracking.tracker import TrackedObjects

_MISSING = BallState(frame_idx=-1, xy=None, source="missing", airborne=False, confidence=0.0)
_CANDIDATE_COLOR = (200, 200, 200)  # BGR grey — deliberately duller than the chosen path
MINIMAP_BALL_BGR = (0, 0, 0)


def draw_candidates(
    frame: np.ndarray, candidates: list[tuple[float, float, float, str]], radius: int = 12
) -> np.ndarray:
    """Draw raw candidates as hollow rings (input never mutated)."""
    scene = frame.copy()
    for x, y, _conf, _source in candidates:
        cv2.circle(scene, (int(round(x)), int(round(y))), radius, _CANDIDATE_COLOR, 1)
    return scene


class DebugRenderer:
    """Boxes, ids and the ball trail."""

    def __init__(
        self,
        teams: dict[int, TrackLabel] | None,
        candidates: dict[int, list[tuple[float, float, float, str]]],
    ) -> None:
        self._tracks = TrackAnnotator(teams=teams)
        self._ball = BallAnnotator()
        self._candidates = candidates

    def draw(
        self, idx: int, frame: np.ndarray, tracks: TrackedObjects | None, ball: BallState | None
    ) -> np.ndarray:
        scene = frame
        if self._candidates:
            scene = draw_candidates(scene, self._candidates.get(idx, []))
        if tracks is not None:
            scene = self._tracks.annotate(scene, tracks)
        # Not covered by the ball pass: no marker, and the trail is held.
        return self._ball.annotate(scene, ball if ball is not None else _MISSING)


def running_share(states: list[PossessionState]) -> dict[int, float | None]:
    """Team A's share of the frames either team has held the ball, up to each frame."""
    shares: dict[int, float | None] = {}
    a = b = 0
    for st in sorted(states, key=lambda s: s.frame_idx):
        a += st.state == "team_a"
        b += st.state == "team_b"
        shares[st.frame_idx] = a / (a + b) if a + b else None
    return shares


def display_ids(tracks_by_idx: dict[int, TrackedObjects]) -> dict[int, str]:
    """Track id -> "1", "2", ... in order of first appearance."""
    names: dict[int, str] = {}
    for idx in sorted(tracks_by_idx):
        for track_id in tracks_by_idx[idx].id:
            names.setdefault(int(track_id), str(len(names) + 1))
    return names


class DeliverableRenderer:
    """Kit-coloured ellipses, a ring round detected balls, the glass minimap and,
    given possession, the running split as a bar above it."""

    def __init__(
        self,
        config: Config,
        teams: dict[int, TrackLabel],
        colours: KitColours,
        positions: list[PitchPosition],
        ball_positions: list[BallPitchState],
        labels: dict[int, str] | None = None,
        possession: list[PossessionState] | None = None,
    ) -> None:
        self._players = KitAnnotator(teams, colours, labels=labels)
        self._shares = running_share(possession) if possession is not None else None
        self._share: float | None = None
        self._teams = teams
        self._colours = colours
        self._spec = config.pitch.to_spec()
        self._geom: Minimap | None = None  # sized from the first frame
        self._positions: dict[int, list[PitchPosition]] = {}
        for p in positions:
            self._positions.setdefault(p.frame_idx, []).append(p)
        self._ball = {b.frame_idx: b for b in ball_positions}
        self._speeds = ball_speeds(ball_positions, config.projection.fps)
        self._max_speed = config.ball.airborne_speed_ms

    def draw(
        self, idx: int, frame: np.ndarray, tracks: TrackedObjects | None, ball: BallState | None
    ) -> np.ndarray:
        if self._geom is None:
            h, w = frame.shape[:2]
            self._geom = minimap_geometry((w, h), self._spec)
        scene = self._players.annotate(frame, tracks) if tracks is not None else frame
        if ball is not None and ball.source == "detected" and ball.xy is not None:
            scene = draw_ball_marker(scene, ball.xy)
        player_r = max(3, round(0.8 * self._geom.scale))
        dots = [
            Dot(p.xy_m, self._colours.for_label(self._teams.get(p.track_id)), player_r, False)
            for p in self._positions.get(idx, [])
        ]
        b = self._ball.get(idx)
        if b is not None and ball_verified(b, self._speeds.get(idx), self._max_speed):
            ball_r = max(2, round(0.5 * self._geom.scale))
            dots.append(Dot(b.xy_m, MINIMAP_BALL_BGR, ball_r, outline=True))
        scene = draw_minimap(scene, dots, self._spec, self._geom)
        if self._shares is None:
            return scene
        self._share = self._shares.get(idx, self._share)
        return draw_possession_bar(scene, self._geom, self._share, self._colours.team)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Draw player + ball passes onto the footage.")
    p.add_argument("--config", type=Path, help="YAML config (must match both passes).")
    p.add_argument("--video", type=Path, help="Input video (overrides config).")
    p.add_argument("--tracks", type=Path, default=None, help="MOT file from the player pass.")
    p.add_argument("--ball", type=Path, default=None, help="Ball CSV from the ball pass.")
    p.add_argument(
        "--candidates", type=Path, default=None, help="Candidates CSV — draws raw candidate rings."
    )
    p.add_argument(
        "--teams",
        type=Path,
        default=None,
        help="teams.csv from the tracking run; colours by team/role instead of track id.",
    )
    p.add_argument(
        "--palette",
        type=Path,
        default=None,
        help="palette.json from the tracking run; switches to the deliverable render.",
    )
    p.add_argument("--positions", type=Path, default=None, help="positions.csv (deliverable).")
    p.add_argument(
        "--ball-positions", type=Path, default=None, help="ball_positions.csv (deliverable)."
    )
    p.add_argument(
        "--possession",
        type=Path,
        default=None,
        help="possession.csv (deliverable): the running split as a bar above the minimap.",
    )
    p.add_argument(
        "--no-ids",
        action="store_true",
        help="Deliverable only: drop the number under each ellipse.",
    )
    p.add_argument(
        "--output",
        type=Path,
        help="Output MP4 (default: <output_dir>/demo.mp4, or annotated.mp4 without --palette).",
    )
    p.add_argument("--max-frames", type=int, default=None, help="Stop after this many frames.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config) if args.config else Config()
    if args.video:
        config.io.input_video = args.video
    if config.io.input_video is None:
        sys.exit("No input video — pass --video or set io.input_video in the config.")
    deliverable = args.palette is not None
    if deliverable and not (args.tracks and args.teams and args.positions and args.ball_positions):
        sys.exit(
            "--palette renders the deliverable and needs --tracks, --teams, --positions "
            "and --ball-positions."
        )
    if not args.tracks and not args.ball:
        sys.exit("Nothing to draw — pass --tracks and/or --ball.")

    video = Path(config.io.input_video)
    tracks_by_idx = read_mot(args.tracks) if args.tracks else {}
    ball_by_idx = read_ball_csv(args.ball) if args.ball else {}
    if args.tracks:
        print(f"Loaded tracks on {len(tracks_by_idx)} frames from {args.tracks}")
    if args.ball:
        print(f"Loaded {len(ball_by_idx)} ball states from {args.ball}")
    teams = read_teams_csv(args.teams) if args.teams else None

    if deliverable:
        require_contiguous_frames(config)
        check_projection_fps(config)
        colours = KitColours.from_palette(read_palette(require_file(args.palette, "palette")))
        positions = read_positions_csv(require_file(args.positions, "positions"))
        ball_positions = read_ball_positions_csv(
            require_file(args.ball_positions, "ball-positions")
        )
        labels = None if args.no_ids else display_ids(tracks_by_idx)
        possession = (
            read_possession_csv(require_file(args.possession, "possession"))
            if args.possession
            else None
        )
        renderer = DeliverableRenderer(
            config, teams or {}, colours, positions, ball_positions, labels, possession
        )
        default_name = "demo.mp4"
    else:
        cands_by_idx = read_candidates_csv(args.candidates) if args.candidates else {}
        if teams is not None:
            print(f"Colouring by team/role from {args.teams} ({len(teams)} tracks)")
        renderer = DebugRenderer(teams, cands_by_idx)
        default_name = "annotated.mp4"

    crop = resolve_crop(config.io, video)
    reader = VideoReader(
        video,
        crop=crop,
        start=config.io.frame_start,
        stop=config.io.frame_stop,
        stride=config.io.frame_stride,
    )
    fps = max(reader.metadata.fps / config.io.frame_stride, 1.0)
    out_path = args.output or (config.io.output_dir / default_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer: cv2.VideoWriter | None = None
    by_source: Counter[str] = Counter()
    processed = 0
    overlap = 0
    try:
        for idx, frame in reader:
            tracks = tracks_by_idx.get(idx)
            state = ball_by_idx.get(idx)
            scene = renderer.draw(idx, frame, tracks, state)
            if state is not None:
                by_source[state.source] += 1
            if tracks is not None and state is not None and state.xy is not None:
                overlap += 1

            if writer is None:
                h, w = scene.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
                if not writer.isOpened():
                    sys.exit(
                        f"cv2.VideoWriter failed to open {out_path}; check the mp4v codec is "
                        "available and the directory is writable."
                    )
            writer.write(scene)
            processed += 1
            if args.max_frames is not None and processed >= args.max_frames:
                break
    finally:
        reader.close()
        if writer is not None:
            writer.release()

    print(f"Wrote {processed} frames to {out_path}")
    print(f"Ball frames by source: {dict(by_source) or '(none)'}")
    print(f"Frames with both a ball position and player tracks: {overlap}")
    if args.tracks and args.ball and overlap == 0 and processed:
        print(
            "WARNING: the two passes never overlap — check they used the same "
            "--config (crop / frame range / stride) so frame indices align."
        )


if __name__ == "__main__":
    main()
