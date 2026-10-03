"""Run pitch calibration over a clip and cache the H/GMC stream.

Seeds H from the hand-labelled anchors ``scripts/annotate_pitch_points.py`` writes,
then fuses forward and backward flow propagation between them.

Writes:

- ``<output_dir>/calibration.npz``   — per-frame H + GMC + flags
- ``<output_dir>/calibration.mp4``   — pitch-line overlay + radar side strip
- ``<output_dir>/mask_debug/*.jpg``  — frame × pitch-mask blend, every N frames
- ``<output_dir>/calibration_metrics.json`` — coverage / solve-quality summary

    uv run python scripts/calibrate_video.py --config configs/<clip>.yaml \\
        --anchors data/annotations/pitch_points_<clip>.json \\
        --output-dir outputs/<clip>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from football_tracker.config import Config
from football_tracker.homography.anchors import load_anchors
from football_tracker.homography.fuse import RANSAC_THRESH_PX, fused_calibration_stream
from football_tracker.homography.mask import pitch_mask
from football_tracker.homography.pitch import PitchSpec
from football_tracker.homography.propagate import FrameMotion, MotionEstimator
from football_tracker.homography.provenance import check_anchor_coverage, provenance_block
from football_tracker.homography.quality import anchor_quality, stream_jitter
from football_tracker.homography.record import FrameCalibration
from football_tracker.homography.render import draw_pitch_overlay, render_radar
from football_tracker.homography.store import save_calibration
from football_tracker.io.video import CropBox, VideoReader, probe_video, resolve_crop

SOURCE_COLORS = {"solved": (80, 220, 80), "propagated": (60, 200, 255), "none": (60, 60, 255)}

# Where an anchor stops being worth trusting. Self-fit error roughly doubles below
# this: 71-85% inliers lands near 0.28 m, ~50% near 0.71 m, and the fused stream can
# be no better than the anchors it is built from.
MIN_ANCHOR_INLIER_RATIO = 0.7


def _scan_motion(
    video: Path, crop: CropBox, config: Config, max_frames: int | None
) -> tuple[dict[int, FrameMotion], list[int], tuple[int, int] | None]:
    """Decode once for global motion only, the first half of the fused two-pass."""
    reader = VideoReader(
        video,
        crop=crop,
        start=config.io.frame_start,
        stop=config.io.frame_stop,
        stride=config.io.frame_stride,
    )
    estimator = MotionEstimator()
    motions: dict[int, FrameMotion] = {}
    frames: list[int] = []
    shape: tuple[int, int] | None = None
    try:
        for idx, frame in reader:
            shape = frame.shape[:2]
            motion = estimator.step(frame)
            if motion is not None:
                motions[idx] = motion
            frames.append(idx)
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        reader.close()
    return motions, frames, shape


def _annotate(frame: np.ndarray, calib: FrameCalibration, spec: PitchSpec) -> np.ndarray:
    """Overlay projected pitch lines + a radar strip + a status HUD."""
    color = SOURCE_COLORS[calib.source]
    out = draw_pitch_overlay(frame, calib.H, spec, color=color) if calib.H is not None else frame
    radar = render_radar(spec, calib.H, frame.shape[:2], scale=4.0)
    target_w = out.shape[1] // 4
    scale = target_w / radar.shape[1]
    radar = cv2.resize(radar, (target_w, int(radar.shape[0] * scale)))
    h, w = radar.shape[:2]
    out[-h - 8 : -8, 8 : 8 + w] = radar
    hud = (
        f"frame {calib.frame_idx}  {calib.source}"
        f"  inliers {calib.inliers}  err {calib.reproj_err_px:.1f}px"
        + ("  LOW-CONF" if calib.low_confidence else "")
    )
    cv2.putText(out, hud, (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5)
    cv2.putText(out, hud, (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Cache per-frame pitch calibration for a clip.")
    p.add_argument("--config", type=Path, help="YAML config (defaults used if omitted).")
    p.add_argument("--video", type=Path, help="Input video (overrides config).")
    p.add_argument("--output-dir", type=Path, help="Output directory (overrides config).")
    p.add_argument("--stride", type=int, help="Process every Nth frame (overrides config).")
    p.add_argument("--max-frames", type=int, default=None, help="Stop after N processed frames.")
    p.add_argument(
        "--mask-every", type=int, default=60, help="Write a mask-debug frame every N frames."
    )
    p.add_argument(
        "--anchors",
        type=Path,
        required=True,
        help="Hand-labelled pitch_points.json from scripts/annotate_pitch_points.py",
    )
    p.add_argument(
        "--max-anchor-gap",
        type=int,
        default=150,
        help="Propagation budget between anchors (frames). Flow drifts ~0.1 m over 150 "
        "frames; wider gaps risk failing the calibration gate.",
    )
    p.add_argument(
        "--allow-anchor-gaps",
        action="store_true",
        help="Accept gaps wider than --max-anchor-gap (and an unanchored tail) as a "
        "warning instead of an error. Does NOT waive a wrong-clip anchors file, which "
        "is provable and always fatal.",
    )
    p.add_argument(
        "--anchor-report",
        action="store_true",
        help="Print inlier ratio, residual and pitch reach for every anchor, not only "
        "those below the warning threshold.",
    )
    args = p.parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()
    if args.video:
        config.io.input_video = args.video
    if args.output_dir:
        config.io.output_dir = args.output_dir
    if args.stride:
        config.io.frame_stride = args.stride
    if config.io.input_video is None:
        raise SystemExit("No input video — pass --video or set io.input_video in the config.")

    video = Path(config.io.input_video)
    meta = probe_video(video)

    # Provenance guard, before any decoding: a calibration stream describes neither the
    # video nor the labels it came from, so a wrong pairing is invisible downstream until
    # a metric collapses somewhere far away.
    anchors = load_anchors(args.anchors)
    coverage = check_anchor_coverage(
        sorted(anchors),
        frame_count=meta.frame_count,
        max_gap_frames=args.max_anchor_gap,
    )
    header = f"anchor coverage problems for {video} vs {args.anchors}:"
    if coverage.fatal:
        body = "\n".join(f"  - {p}" for p in coverage.fatal)
        raise SystemExit(
            f"{header}\n{body}\n"
            "This is not a tuning threshold and no flag overrides it: pass the "
            "anchors labelled on THIS clip."
        )
    if coverage.advisory:
        body = "\n".join(f"  - {p}" for p in coverage.advisory)
        if args.allow_anchor_gaps:
            print(f"warning: {header}\n{body}", file=sys.stderr)
        else:
            raise SystemExit(
                f"{header}\n{body}\n"
                "Add anchors with scripts/annotate_pitch_points.py, or pass "
                "--allow-anchor-gaps to accept the wider propagation budget."
            )

    spec = config.pitch.to_spec()
    # Per-anchor self-consistency, before the two decode passes
    qualities = {
        frame: anchor_quality(*anchors[frame], spec, ransac_thresh_px=RANSAC_THRESH_PX)
        for frame in sorted(anchors)
    }
    print(f"{len(anchors)} anchors from {args.anchors}")
    if args.anchor_report:
        for frame, q in qualities.items():
            print(
                f"  frame {frame:>6}  {q.n_inliers}/{q.n_points} inliers "
                f"({q.inlier_ratio:.0%})  {q.median_residual_m:.2f} m  reach {q.end_reach}"
            )
    poor = {f: q for f, q in qualities.items() if q.inlier_ratio < MIN_ANCHOR_INLIER_RATIO}
    if poor:
        print(
            f"warning: {len(poor)} of {len(anchors)} anchors are below "
            f"{MIN_ANCHOR_INLIER_RATIO:.0%} inliers; the fused stream lands at its "
            "anchors' own quality:",
            file=sys.stderr,
        )
        for frame, q in poor.items():
            revisit = f"; revisit points {q.outlier_indices[:3]}" if q.outlier_indices else ""
            print(
                f"  frame {frame}: {q.n_inliers}/{q.n_points} inliers "
                f"({q.inlier_ratio:.0%}), {q.median_residual_m:.2f} m{revisit}",
                file=sys.stderr,
            )

    crop = resolve_crop(config.io, video)
    out_dir = Path(config.io.output_dir)
    mask_dir = out_dir / "mask_debug"
    mask_dir.mkdir(parents=True, exist_ok=True)

    # The clip is calibrated up front as a whole -- a first decode pass measures
    # camera motion, the fusion turns that plus the anchors into one H per frame --
    # and the render loop below only looks the answer up.
    motions, scanned, shape = _scan_motion(video, crop, config, args.max_frames)
    fused = {
        c.frame_idx: c
        for c in fused_calibration_stream(anchors, motions, scanned, spec=spec, frame_shape=shape)
    }
    print(f"Bidirectional fusion over {len(scanned)} frames from {len(anchors)} anchors")

    reader = VideoReader(
        video,
        crop=crop,
        start=config.io.frame_start,
        stop=config.io.frame_stop,
        stride=config.io.frame_stride,
    )
    fps = max(reader.metadata.fps / config.io.frame_stride, 1.0)
    writer: cv2.VideoWriter | None = None
    stream: list[FrameCalibration] = []

    try:
        for idx, frame in reader:
            calib = fused[idx]
            stream.append(calib)
            annotated = _annotate(frame, calib, spec)
            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(
                    str(out_dir / "calibration.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                )
            writer.write(annotated)
            if calib.H is not None and len(stream) % args.mask_every == 0:
                mask = pitch_mask(
                    calib.H, frame.shape[:2], spec, margin_m=config.homography.mask_margin_m
                )
                blend = frame.copy()
                blend[mask == 0] = blend[mask == 0] // 3
                cv2.imwrite(str(mask_dir / f"mask_{idx:05d}.jpg"), blend)
            if args.max_frames is not None and len(stream) >= args.max_frames:
                break
    finally:
        reader.close()
        if writer is not None:
            writer.release()

    save_calibration(out_dir / "calibration.npz", stream, frame_shape=shape)

    sources = [c.source for c in stream]
    solved = [c for c in stream if c.source == "solved"]
    # Accuracy and continuity are different failures: the held-out gate catches a map
    # that is wrong, this catches one that jumps. Averaged positions survive a seam;
    # speed and distance covered do not.
    jitter = stream_jitter(stream, spec)
    metrics = {
        # Provenance first: which clip, which labels, and the anchor layout they imply.
        **provenance_block(video, args.anchors, meta.frame_count, sorted(anchors)),
        "crop": vars(crop),
        "anchor_quality": {
            str(frame): {
                "inlier_ratio": round(q.inlier_ratio, 3),
                "median_residual_m": round(q.median_residual_m, 3),
                "end_reach": q.end_reach,
                "outlier_indices": q.outlier_indices,
            }
            for frame, q in qualities.items()
        },
        "frames": len(stream),
        "solved": sources.count("solved"),
        "propagated": sources.count("propagated"),
        "uncalibrated": sources.count("none"),
        "low_confidence": sum(c.low_confidence for c in stream),
        "median_solve_inliers": float(np.median([c.inliers for c in solved])) if solved else None,
        "median_solve_reproj_err_px": (
            float(np.median([c.reproj_err_px for c in solved])) if solved else None
        ),
        "jitter_px": (
            {
                "median": round(jitter.median_px, 3),
                "p90": round(jitter.p90_px, 3),
                "max": round(jitter.max_px, 3),
                "frames_above_noise": len(jitter.worst_frames),
                "worst_frames": jitter.worst_frames[:5],
            }
            if jitter.n_pairs
            else None
        ),
    }
    (out_dir / "calibration_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"\nWrote {out_dir}/calibration.npz, calibration.mp4, mask_debug/, metrics json")


if __name__ == "__main__":
    main()
