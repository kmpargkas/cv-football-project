"""Hand-label image↔pitch correspondences

Usage (opens an OpenCV window; needs a display):

    uv run python scripts/annotate_pitch_points.py --video data/raw/<clip>.mp4 \\
        --frames 0,100,200,300,400,500,600,700

Keys:
    click = place the current target   s = skip it        u = back one target
    n     = next frame                 q = save & quit

Targets are ordered so the most useful crossings come first.

Output JSON: ``data/annotations/pitch_points.json`` (image coords are in the
*cropped* frame, matching the H convention).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from football_tracker.config import Config
from football_tracker.homography.pitch import PHANTOM_VERTICES, PitchSpec
from football_tracker.io.video import CropBox, VideoReader, resolve_crop
from football_tracker.pipeline.onboard import suggest_anchor_frames

WINDOW = "annotate pitch points"

VERTEX_NAMES: tuple[str, ...] = (
    "left corner - FAR side",
    "left penalty box x goal line - FAR",
    "left goal box x goal line - FAR",
    "left goal box x goal line - NEAR",
    "left penalty box x goal line - NEAR",
    "left corner - NEAR side",
    "left goal box outer corner - FAR",
    "left goal box outer corner - NEAR",
    "left penalty spot",
    "left penalty box outer corner - FAR",
    "left penalty box x goal-box line - FAR",
    "left penalty box x goal-box line - NEAR",
    "left penalty box outer corner - NEAR",
    "halfway line x FAR touchline",
    "halfway line x centre circle - FAR",
    "halfway line x centre circle - NEAR",
    "halfway line x NEAR touchline",
    "right penalty box outer corner - FAR",
    "right penalty box x goal-box line - FAR",
    "right penalty box x goal-box line - NEAR",
    "right penalty box outer corner - NEAR",
    "right penalty spot",
    "right goal box outer corner - FAR",
    "right goal box outer corner - NEAR",
    "right corner - FAR side",
    "right penalty box x goal line - FAR",
    "right goal box x goal line - FAR",
    "right goal box x goal line - NEAR",
    "right penalty box x goal line - NEAR",
    "right corner - NEAR side",
    "centre circle left extreme (imprecise - prefer skipping)",
    "centre circle right extreme (imprecise - prefer skipping)",
)

TARGET_ORDER: tuple[int, ...] = (
    13,
    16,
    14,
    15,
    17,
    20,
    21,
    22,
    23,
    25,
    26,
    27,
    28,
    24,
    29,
    9,
    12,
    8,
    6,
    7,
    1,
    2,
    3,
    4,
    0,
    5,
    30,
    31,
)
assert not set(TARGET_ORDER) & set(PHANTOM_VERTICES), "phantoms are never offered"


def _pitch_reference(spec: PitchSpec, target: int | None, width: int = 520) -> np.ndarray:
    """Small top-down pitch with every vertex dotted and ``target`` highlighted."""
    pad = 18
    scale = (width - 2 * pad) / spec.length
    height = int(spec.width * scale) + 2 * pad
    canvas = np.full((height, width, 3), (45, 85, 45), np.uint8)

    def to_px(p: np.ndarray) -> tuple[int, int]:
        return (int(p[0] * scale + pad), int(p[1] * scale + pad))

    for segment in spec.line_segments():
        poly = np.array([to_px(p) for p in segment], np.int32)
        cv2.polylines(canvas, [poly], False, (215,) * 3, 1)
    for kp in spec.keypoints:
        cv2.circle(canvas, to_px(kp), 3, (0, 190, 255), -1)
    if target is not None:
        centre = to_px(spec.keypoints[target])
        cv2.circle(canvas, centre, 13, (0, 0, 255), 2)
        cv2.drawMarker(canvas, centre, (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
    cv2.putText(canvas, "FAR (y=0)", (pad, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (215,) * 3, 1)
    cv2.putText(
        canvas, "NEAR (y=68)", (pad, height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (215,) * 3, 1
    )
    return canvas


def _hud(disp: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    """Draw shadowed HUD lines at the top-left of the display image."""
    for i, (text, color) in enumerate(lines):
        origin = (14, 34 + i * 34)
        cv2.putText(disp, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 5)
        cv2.putText(disp, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)


def _grab_frames(video: Path, crop: CropBox, indices: list[int]) -> dict[int, np.ndarray]:
    frames: dict[int, np.ndarray] = {}
    with VideoReader(video, crop=crop) as reader:
        wanted = set(indices)
        for idx, frame in reader:
            if idx in wanted:
                frames[idx] = frame
                wanted.discard(idx)
                if not wanted:
                    break
    return frames


def _annotate_frame(
    frame: np.ndarray,
    idx: int,
    spec: PitchSpec,
    display_width: int,
) -> list[dict[str, object]]:
    """Guided labelling of one frame; returns its correspondence list."""
    targets = TARGET_ORDER
    scale = min(1.0, display_width / frame.shape[1])
    points: list[dict[str, object]] = []
    clicked: list[tuple[float, float]] = []
    cursor = 0

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((x / scale, y / scale))

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    print(f"\n=== frame {idx}: click=place  s=skip  u=back  n=next  q=save&quit ===")

    while True:
        target = targets[cursor] if cursor < len(targets) else None
        disp = cv2.resize(frame, None, fx=scale, fy=scale)
        for p in points:
            cx, cy = (int(v * scale) for v in p["image"])
            cv2.drawMarker(disp, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
            cv2.putText(
                disp,
                f"{p['pitch'][0]:.1f},{p['pitch'][1]:.1f}",
                (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )
        if target is not None:
            head = (f"CLICK v{target + 1}: {VERTEX_NAMES[target]}", (80, 255, 80))
        else:
            head = ("all targets offered - n for next frame", (200, 200, 200))
        _hud(
            disp,
            [
                head,
                (
                    f"frame {idx}   placed {len(points)}   target {min(cursor + 1, len(targets))}"
                    f"/{len(targets)}   [s]kip [u]ndo [n]ext [q]uit",
                    (235, 235, 235),
                ),
            ],
        )
        panel = _pitch_reference(spec, target)
        ph, pw = panel.shape[:2]
        disp[disp.shape[0] - ph - 10 : -10, disp.shape[1] - pw - 10 : -10] = panel
        cv2.imshow(WINDOW, disp)
        key = cv2.waitKey(30) & 0xFF

        if clicked:
            image_pt = clicked.pop(0)  # FIFO: queued clicks stay in click order
            if target is None:
                print("  ! every target has been offered - press n for the next frame")
                continue
            pitch_pt = tuple(spec.keypoints[target])
            points.append(
                {
                    "image": list(image_pt),
                    "pitch": [float(v) for v in pitch_pt],
                    "vertex": target + 1,
                }
            )
            cursor += 1
            print(f"  placed ({image_pt[0]:.0f},{image_pt[1]:.0f}) -> pitch {pitch_pt}")
        elif key == ord("s") and target is not None:
            cursor += 1
        elif key == ord("u") and cursor > 0:
            cursor -= 1
            if points and points[-1]["vertex"] == targets[cursor] + 1:
                points.pop()
        elif key == ord("n"):
            return points
        elif key == ord("q"):
            raise KeyboardInterrupt


def main() -> None:
    p = argparse.ArgumentParser(description="Label image↔pitch correspondences for eval.")
    p.add_argument("--config", type=Path, help="YAML config (defaults used if omitted).")
    p.add_argument("--video", type=Path, help="Input video (overrides config).")
    p.add_argument("--frames", type=str, default=None, help="Comma-separated frame indices.")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/annotations/pitch_points.json"),
        help="Output JSON path (existing file is extended/overwritten per frame).",
    )
    p.add_argument("--display-width", type=int, default=1600, help="Max on-screen width (px).")
    args = p.parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()
    if args.video:
        config.io.input_video = args.video
    if config.io.input_video is None:
        raise SystemExit("No input video — pass --video or set io.input_video in the config.")
    video = Path(config.io.input_video)
    crop = resolve_crop(config.io, video)
    spec = config.pitch.to_spec()

    if args.frames:
        indices = [int(v) for v in args.frames.split(",")]
    else:
        with VideoReader(video, crop=crop) as reader:
            total = reader.metadata.frame_count
        indices = suggest_anchor_frames(total)
        print(f"Labelling {len(indices)} frames: {','.join(str(i) for i in indices)}")

    frames = _grab_frames(video, crop, indices)
    existing = json.loads(args.output.read_text()) if args.output.exists() else {}
    labeled: dict[str, list] = existing.get("frames", {})
    try:
        for idx in indices:
            if idx not in frames:
                print(f"frame {idx}: not decodable, skipped")
                continue
            points = _annotate_frame(frames[idx], idx, spec, args.display_width)
            if points:
                labeled[str(idx)] = points
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"video": str(video), "crop": vars(crop), "frames": labeled}, indent=2)
    )
    counts = {k: len(v) for k, v in labeled.items()}
    print(f"\nSaved {sum(counts.values())} points over {len(labeled)} frames -> {args.output}")
    print(f"Per frame: {counts}")


if __name__ == "__main__":
    main()
