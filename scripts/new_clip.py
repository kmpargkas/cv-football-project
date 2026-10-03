"""Onboard a new tactical clip: probe it, scaffold a config, plan its anchors.

    uv run python scripts/new_clip.py --video data/raw/newmatch.mp4

Prints the anchor frames to label, writes configs/<stem>.yaml, and then tells you
exactly which two commands to run. Re-running it is safe: it never overwrites an
existing config, and it reports what is still missing."""

from __future__ import annotations

import argparse
from pathlib import Path

from football_tracker.config import IOConfig
from football_tracker.io.video import probe_video, resolve_crop
from football_tracker.pipeline.onboard import (
    onboarding_checklist,
    scaffold_config,
    suggest_anchor_frames,
)

# run_pipeline derives this path.
ANCHORS_DIR = Path("data/annotations")


def main() -> None:
    ap = argparse.ArgumentParser(description="Scaffold a new clip: config + anchor plan.")
    ap.add_argument("--video", type=Path, required=True, help="Clip to onboard.")
    ap.add_argument(
        "--max-gap",
        type=int,
        default=100,
        help="Largest allowed gap between anchors. The 100 default is the measured",
    )
    ap.add_argument(
        "--config-dir", type=Path, default=Path("configs"), help="Where to write the config."
    )
    args = ap.parse_args()

    if not args.video.exists():
        raise SystemExit(f"no such video: {args.video}")

    meta = probe_video(args.video)
    print(
        f"{args.video.name}: {meta.width}x{meta.height}, {meta.fps:.6g} fps, "
        f"{meta.frame_count} frames ({meta.duration:.1f} s)"
    )

    # IOConfig() defaults are exactly what scaffold_config writes, so the bars
    # reported here are the ones the run will actually crop.
    crop = resolve_crop(IOConfig(), args.video)
    if crop.is_empty:
        print("letterbox: none detected")
    else:
        print(
            f"letterbox: top={crop.top} bottom={crop.bottom} left={crop.left} "
            f"right={crop.right} — auto_letterbox is on, so the run will crop this. "
            f"Check it looks right before labelling anchors against the cropped frame."
        )

    frames = suggest_anchor_frames(meta.frame_count, max_gap=args.max_gap)
    print(f"\nanchors: {len(frames)} frames, every gap <= {args.max_gap}")
    frame_list = ",".join(str(f) for f in frames)
    print(frame_list)

    stem = args.video.stem
    config_path = args.config_dir / f"{stem}.yaml"
    anchors_path = ANCHORS_DIR / f"pitch_points_{stem}.json"
    if config_path.exists():
        print(
            f"\n{config_path} already exists — not overwriting. To see what a fresh "
            f"scaffold would contain:\n"
            f'  uv run python -c "from football_tracker.pipeline.onboard import '
            f"scaffold_config; print(scaffold_config('{args.video}', {meta.fps}))\" "
            f"| diff {config_path} -"
        )
    else:
        args.config_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(scaffold_config(args.video, meta.fps))
        print(f"\nwrote {config_path}")

    print(
        f"\nnext:\n"
        f"  uv run python scripts/annotate_pitch_points.py --config {config_path} \\\n"
        f"      --frames {frame_list} \\\n"
        f"      --output {anchors_path}\n"
        f"  uv run python scripts/run_pipeline.py --config {config_path}"
    )

    missing = onboarding_checklist(anchors_path, config_path)
    if missing:
        print("\nstill missing:")
        for item in missing:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
