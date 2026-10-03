"""Offline tracklet stitching over a completed tracking run.

Reads ``tracks.txt`` plus its ``teams.csv`` and the calibration, rejoins the fragments
one person was split into, and writes a separate ``tracks.stitched.txt`` rather than
rewriting the input.

    uv run python scripts/stitch_tracks.py --config <clip>.yaml \\
        --tracks outputs/<clip>/tracks.txt --teams outputs/<clip>/teams.csv \\
        --calibration outputs/<clip>/calibration.npz --video <clip>.mp4 \\
        --out outputs/<clip>/tracks.stitched.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from football_tracker.config import Config
from football_tracker.homography.solve import project
from football_tracker.homography.store import load_calibration
from football_tracker.io.video import probe_video
from football_tracker.reid.stitch import apply_remap, build_tracklets, roster_overflow, solve_links
from football_tracker.tracking.artifacts import mot_row, read_mot, read_teams_csv


def main() -> None:
    p = argparse.ArgumentParser(description="Offline tracklet stitching.")
    p.add_argument("--config", type=Path, help="YAML config (defaults used if omitted).")
    p.add_argument("--tracks", type=Path, required=True, help="MOT file from a tracking pass.")
    p.add_argument("--teams", type=Path, required=True, help="teams.csv from the same pass.")
    p.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="calibration.npz (from the calibration stage).",
    )
    p.add_argument("--video", type=Path, required=True, help="Clip, for the frame rate.")
    p.add_argument("--out", type=Path, required=True, help="Where to write tracks.stitched.txt.")
    args = p.parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()
    cfg = config.stitch
    fps = probe_video(args.video).fps

    tracks_by_frame = read_mot(args.tracks)
    labels = read_teams_csv(args.teams)
    calib = {c.frame_idx: c for c in load_calibration(args.calibration)}

    # Foot point through the per-frame homography. Frames without a usable calibration
    # contribute nothing rather than a guessed position.
    positions: dict[int, dict[int, np.ndarray]] = {}
    for frame, tracks in tracks_by_frame.items():
        c = calib.get(frame)
        if c is None or c.H is None or c.low_confidence or not len(tracks):
            continue
        foot = np.column_stack([(tracks.xyxy[:, 0] + tracks.xyxy[:, 2]) / 2.0, tracks.xyxy[:, 3]])
        pitch = project(c.H, foot)
        positions[frame] = {int(t): pitch[i] for i, t in enumerate(tracks.id)}

    tracklets = build_tracklets(positions, labels, fps=fps, velocity_window_s=cfg.velocity_window_s)
    remap = solve_links(tracklets, fps=fps, cfg=cfg)

    before = {int(t) for tracks in tracks_by_frame.values() for t in tracks.id}
    merged = {k: v for k, v in remap.items() if k != v}
    after = {remap.get(t, t) for t in before}

    stitched = apply_remap(tracks_by_frame, remap)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        mot_row(frame, int(tid), int(cls), box)
        for frame in sorted(stitched)
        for tid, cls, box in zip(
            stitched[frame].id, stitched[frame].class_id, stitched[frame].xyxy, strict=True
        )
    ]
    args.out.write_text("\n".join(rows) + "\n")

    overflow = roster_overflow(tracklets, remap, cfg.roster_outfield)
    print(f"Tracklets: {len(tracklets)} ({len(before)} ids in the input)")
    print(f"Merges:    {len(merged)} ids remapped -> {len(after)} identities")
    for source, target in sorted(merged.items()):
        print(f"    {source} -> {target}")
    print(f"Wrote {len(rows)} rows to {args.out}")
    if overflow:
        print(f"Roster overflow (under-merged, by role): {overflow}")
    else:
        print(f"No role exceeds {cfg.roster_outfield} simultaneous identities.")


if __name__ == "__main__":
    main()
