"""Detect and track over a clip; write the tracks, teams and ball CSVs.

The pipeline runs this twice. The players pass tracks players, goalkeepers and
referees with BoT-SORT, assigns each track a team and role from torso colour, and
writes ``tracks.txt`` + ``teams.csv`` + ``palette.json``. The ball pass collects
per-frame ball candidates, solves the trajectory offline, and writes ``ball.csv``.

    uv run python scripts/track_video.py --config <clip>.yaml --no-ball --teams \\
        --mot-out outputs/<clip>/tracks.txt --teams-out outputs/<clip>/teams.csv
    uv run python scripts/track_video.py --config <clip>.yaml --ball \\
        --weights weights/ball.pt --ball-out outputs/<clip>/ball.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from football_tracker.config import Config, TeamIDConfig
from football_tracker.detection.interface import ObjectClass
from football_tracker.detection.yolo import YOLODetector
from football_tracker.homography.record import FrameCalibration
from football_tracker.homography.solve import project
from football_tracker.homography.store import load_calibration
from football_tracker.io.video import VideoReader, resolve_crop
from football_tracker.reid.appearance import torso_descriptor
from football_tracker.reid.teamid import (
    TeamAssigner,
    build_observations,
    canonicalise_palette,
    describe_team_order,
    fit_palette,
)
from football_tracker.tracking.artifacts import (
    mot_row,
    read_palette,
    write_ball_csv,
    write_candidates_csv,
    write_palette,
    write_teams_csv,
)
from football_tracker.tracking.ball import BallCandidate, BallObserver
from football_tracker.tracking.ball_trajectory import build_trajectory
from football_tracker.tracking.tracker import PlayerTracker, TrackedObjects


def _accumulate_teamid(
    frame: np.ndarray,
    tracks: TrackedObjects,
    calib: FrameCalibration | None,
    cfg: TeamIDConfig,
    lab_samples: dict[int, list[np.ndarray]],
    pitch_positions: dict[int, list[np.ndarray]],
    class_counts: dict[int, dict[int, int]],
    track_frames: dict[int, int],
) -> None:
    """Fold one frame's tracks into the TeamID accumulators.

    Pitch positions are only recorded when the frame has a usable homography, so a
    track with none simply gets no goal-zone evidence — which correctly stops it
    claiming goalkeeper on an absence of information rather than on evidence.
    """
    usable = calib is not None and calib.H is not None and not calib.low_confidence
    foot_points = np.column_stack(
        [(tracks.xyxy[:, 0] + tracks.xyxy[:, 2]) / 2.0, tracks.xyxy[:, 3]]
    )
    projected = project(calib.H, foot_points) if usable else None

    for i, (track_id, class_id) in enumerate(zip(tracks.id, tracks.class_id, strict=True)):
        track_id = int(track_id)
        track_frames[track_id] = track_frames.get(track_id, 0) + 1
        counts = class_counts.setdefault(track_id, {})
        counts[int(class_id)] = counts.get(int(class_id), 0) + 1
        if projected is not None:
            pitch_positions.setdefault(track_id, []).append(projected[i])
        sample = torso_descriptor(frame, tracks.xyxy[i], cfg)
        lab_samples.setdefault(track_id, [])
        if sample is not None:
            lab_samples[track_id].append(sample)


def main() -> None:
    p = argparse.ArgumentParser(description="Detect + track players and ball over a clip.")
    p.add_argument("--config", type=Path, help="YAML config (defaults used if omitted).")
    p.add_argument("--video", type=Path, help="Input video (overrides config).")
    p.add_argument("--weights", type=Path, help="Detector weights (overrides config).")
    p.add_argument("--calibration", type=Path, help="calibration.npz (overrides config).")
    p.add_argument("--max-frames", type=int, default=None, help="Stop after this many frames.")
    p.add_argument(
        "--mot-out",
        type=Path,
        default=None,
        help="Write tracks as a MOT-Challenge file.",
    )
    ball_group = p.add_mutually_exclusive_group()
    ball_group.add_argument(
        "--ball", dest="ball", action="store_true", default=None, help="Enable ball tracking."
    )
    ball_group.add_argument(
        "--no-ball", dest="ball", action="store_false", help="Disable ball tracking."
    )
    p.add_argument(
        "--ball-out",
        type=Path,
        default=None,
        help="Write the ball trajectory CSV (frame,x,y,source,airborne,confidence).",
    )
    p.add_argument(
        "--ball-weights",
        type=Path,
        default=None,
        help="Separate detector weights for the ball ROI pass (default: share --weights).",
    )
    p.add_argument(
        "--ball-candidates-out",
        type=Path,
        default=None,
        help="Write raw ball candidates CSV (frame,x,y,confidence,source) for the render's"
        " candidate overlay.",
    )
    team_group = p.add_argument_group("TeamID")
    team_group.add_argument(
        "--teams", action="store_true", help="Assign a team/role to every track."
    )
    team_group.add_argument(
        "--teams-out", type=Path, default=None, help="Write the per-track teams CSV here."
    )
    team_group.add_argument(
        "--palette",
        type=Path,
        default=None,
        help="Load a frozen palette instead of fitting one (reuse across clips of one match).",
    )
    team_group.add_argument(
        "--palette-out", type=Path, default=None, help="Write the fitted palette JSON here."
    )
    args = p.parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()
    if args.video:
        config.io.input_video = args.video
    if args.weights:
        config.detection.weights = args.weights
    if args.calibration:
        config.tracking.calibration_path = args.calibration
    if args.ball is not None:
        config.ball.enabled = args.ball
    if args.ball_weights:
        config.ball.weights = args.ball_weights
    if args.palette:
        config.teamid.palette_path = args.palette
    if config.io.input_video is None:
        raise SystemExit("No input video — pass --video or set io.input_video in the config.")

    # The injected GMC is a per-consecutive-frame affine, so tracking must run at stride 1
    # to keep the cached warp aligned with the frames actually processed.
    if config.io.frame_stride != 1:
        raise SystemExit("Tracking requires frame_stride=1 (GMC is a per-frame affine).")
    if config.tracking.calibration_path is None:
        raise SystemExit("pass --calibration or set tracking.calibration_path in the config.")

    video = Path(config.io.input_video)
    crop = resolve_crop(config.io, video)
    stream = load_calibration(config.tracking.calibration_path)
    spec = config.pitch.to_spec()
    detector = YOLODetector.from_config(config.detection, device=config.device)
    observer = (
        BallObserver.from_config(
            config.ball, config.detection, stream, spec, detector, device=config.device
        )
        if config.ball.enabled
        else None
    )

    reader = VideoReader(video, crop=crop, start=config.io.frame_start, stop=config.io.frame_stop)
    fps = reader.metadata.fps
    tracker = PlayerTracker.from_config(config.tracking, stream, spec, fps=fps)

    ids_by_class: dict[str, set[int]] = {c.name.lower(): set() for c in ObjectClass}
    mot_lines: list[str] | None = [] if args.mot_out else None
    candidates: list[BallCandidate] = []
    first_idx: int | None = None
    last_idx: int | None = None
    processed = 0
    # TeamID accumulation. One decode serves both the palette fit and the assignment:
    # samples are collected here, the palette is fitted afterwards, then the
    # same samples are voted and consolidated per track.
    want_teams = bool(args.teams or args.teams_out or args.palette_out)
    calib_by_idx = {c.frame_idx: c for c in stream}
    lab_samples: dict[int, list[np.ndarray]] = {}
    pitch_positions: dict[int, list[np.ndarray]] = {}
    class_counts: dict[int, dict[int, int]] = {}
    track_frames: dict[int, int] = {}
    try:
        for idx, frame in reader:
            detections = detector.detect(frame)
            tracks = tracker.update(idx, frame, detections)
            for tid, cls in zip(tracks.id, tracks.class_id, strict=True):
                ids_by_class[ObjectClass(int(cls)).name.lower()].add(int(tid))
            if mot_lines is not None:
                for tid, cls, (x1, y1, x2, y2) in zip(
                    tracks.id, tracks.class_id, tracks.xyxy, strict=True
                ):
                    mot_lines.append(mot_row(idx, int(tid), int(cls), np.array([x1, y1, x2, y2])))
            if want_teams and len(tracks):
                _accumulate_teamid(
                    frame,
                    tracks,
                    calib_by_idx.get(idx),
                    config.teamid,
                    lab_samples,
                    pitch_positions,
                    class_counts,
                    track_frames,
                )
            if observer is not None:
                candidates += observer.observe(idx, frame, detections)
            if first_idx is None:
                first_idx = idx
            last_idx = idx
            processed += 1
            if args.max_frames is not None and processed >= args.max_frames:
                break
    finally:
        reader.close()

    if want_teams:
        observations = build_observations(
            lab_samples, pitch_positions, class_counts, track_frames, spec, config.teamid
        )
        if config.teamid.palette_path is not None:
            palette = read_palette(config.teamid.palette_path)
            print(f"TeamID: loaded frozen palette from {config.teamid.palette_path}")
        else:
            palette = fit_palette(observations, config.teamid)
            prov = palette.provenance
            print(
                f"TeamID: fitted on {prov['n_tracks']} eligible tracks, "
                f"silhouette {prov['silhouette']:.3f}, reject_radius "
                f"{palette.reject_radius:.1f}, {prov['n_outliers']} colour outliers"
            )
        # A fitted palette is already canonical; a loaded one may predate the rule.
        palette = canonicalise_palette(palette)
        print(f"TeamID: {describe_team_order(palette)}")
        assigner = TeamAssigner(palette, config.teamid)
        for track_id, labs in lab_samples.items():
            for lab in labs:
                assigner.observe(track_id, lab)
            for _ in range(track_frames.get(track_id, 0) - len(labs)):
                assigner.observe(track_id, None)
        labels = assigner.consolidate(track_frames=track_frames)
        by_role: dict[str, int] = {}
        for label in labels.values():
            by_role[label.role.value] = by_role.get(label.role.value, 0) + 1
        print("TeamID by role:", by_role)
        if args.palette_out:
            write_palette(args.palette_out, palette)
            print(f"Wrote palette to {args.palette_out}")
        if args.teams_out:
            write_teams_csv(args.teams_out, labels)
            print(f"Wrote {len(labels)} track labels to {args.teams_out}")

    if mot_lines is not None:
        args.mot_out.parent.mkdir(parents=True, exist_ok=True)
        args.mot_out.write_text("\n".join(mot_lines) + ("\n" if mot_lines else ""))
        print(f"Wrote {len(mot_lines)} MOT hypothesis rows to {args.mot_out}")

    if observer is not None and first_idx is not None and last_idx is not None:
        # Offline trajectory over the whole processed range.
        states = build_trajectory(
            candidates,
            (first_idx, last_idx),
            stream,
            config.ball,
            fps=fps,
        )
        n_detected = sum(s.source == "detected" for s in states)
        by_source = {
            src: sum(s.source == src for s in states)
            for src in ("detected", "interpolated", "missing")
        }
        if args.ball_out:
            write_ball_csv(args.ball_out, states)
            print(f"Wrote {len(states)} ball states to {args.ball_out}")
        if args.ball_candidates_out:
            write_candidates_csv(args.ball_candidates_out, candidates)
            print(f"Wrote {len(candidates)} ball candidates to {args.ball_candidates_out}")
        print(f"Ball: {len(candidates)} candidates → {n_detected} on the solved path")
        print("Ball frames by source:", by_source, f"airborne={sum(s.airborne for s in states)}")

    total_ids = len(set().union(*ids_by_class.values())) if ids_by_class else 0
    print(f"Tracked {processed} frames; unique IDs={total_ids}")
    print("Unique IDs by class:", {k: len(v) for k, v in ids_by_class.items()})


if __name__ == "__main__":
    main()
