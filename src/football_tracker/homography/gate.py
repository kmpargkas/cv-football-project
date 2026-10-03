"""The pipeline's calibration gate: how accurate is the homography where nobody labelled?

The stream is seeded at every hand-labelled anchor, so scoring it there would only show
how well each fit matches its own clicks. Each anchor is therefore held out in turn:
removed, the homography that frame would have received rebuilt from the surviving anchors
and the cached camera motion, then scored against the labels kept back. A held-out frame
sits at the far end of a propagation gap, so the number is pessimistic by design -- every
genuinely unlabelled frame is closer to a seed than the measurement is.

Reads ``calibration.npz`` and the anchors it was built from; writes
``holdout_metrics.json`` -- overall, per anchor, and per pitch third -- and exits non-zero
when the median exceeds ``--gate-median-m``. Pure numpy on cached data: no video decode,
no model runtime. ``scripts/gate_calibration.py`` is the entry point.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from football_tracker.config import Config
from football_tracker.homography.anchors import load_anchors
from football_tracker.homography.fuse import RANSAC_THRESH_PX, fuse_anchors
from football_tracker.homography.pitch import PitchSpec
from football_tracker.homography.propagate import FrameMotion
from football_tracker.homography.provenance import anchor_fingerprint
from football_tracker.homography.solve import project
from football_tracker.homography.store import load_frame_shape


class CalibrationUnmeasurable(ValueError):
    """The held-out measurement could not be made at all.

    A gate failure like any other -- the caller needs the reason, not a traceback
    pointing at the line that noticed. Subclasses ``ValueError`` so anything already
    catching that keeps working.
    """


def _unmeasurable_reason(
    frame: int, first_anchor: int, last_anchor: int, gmc_by_frame: dict[int, np.ndarray]
) -> str:
    """Name the actual reason the cached motion stream cannot reach ``frame``.

    Three separate calibration settings leave a frame uncovered, and each needs a
    different fix, so the message must not assert one of them. The stream knows its own
    range; comparing it against the anchors says which of the three happened.
    """
    covered = sorted(gmc_by_frame)
    if not covered:
        return "the calibration stream holds no frames at all"
    first, last = covered[0], covered[-1]
    if frame > last:
        return (
            f"anchors run to frame {last_anchor} but the calibration stream ends at "
            f"{last}: the clip was calibrated over a shorter range than it was labelled "
            "(io.frame_stop, or --max-frames). Re-calibrate the labelled range."
        )
    if frame < first:
        return (
            f"anchors start at frame {first_anchor} but the calibration stream starts at "
            f"{first}: those labels precede the calibrated range (io.frame_start)."
        )
    return (
        f"the calibration stream skips frame {frame} inside its own range "
        f"({first}-{last}): it was calibrated with io.frame_stride > 1. Anchors off the "
        "stride grid never seeded that stream either, so re-calibrate at stride 1."
    )


def _check_provenance(calibration: Path, anchors: Path) -> None:
    """Refuse to score a stream against anchors it was not built from.

    The calibrate stage records the anchors' fingerprint beside the stream in
    ``calibration_metrics.json``. A mismatch means this would certify a calibration
    against a different clip's labels -- provable, so never waived. A metrics file that
    is missing or predates the fingerprint is only noted: the check cannot be made, and
    saying so beats silently skipping it.
    """
    metrics = calibration.with_name("calibration_metrics.json")
    recorded = (
        json.loads(metrics.read_text()).get("anchors_sha256_12") if metrics.exists() else None
    )
    if recorded is None:
        print(
            f"note: no anchors fingerprint in {metrics}, so these anchors cannot be checked "
            "against the ones the calibration was built from.",
            file=sys.stderr,
        )
        return
    actual = anchor_fingerprint(anchors)
    if actual != recorded:
        raise CalibrationUnmeasurable(
            f"these anchors ({actual}) are not the ones the calibration was built from "
            f"({recorded}, per {metrics.name}). Re-calibrate with these anchors, or pass "
            "the anchors this stream came from."
        )


def holdout_errors(
    anchors: dict[int, tuple[np.ndarray, np.ndarray]],
    gmc_by_frame: dict[int, np.ndarray],
    ransac_thresh_px: float = RANSAC_THRESH_PX,
    spec: PitchSpec | None = None,
    frame_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Score each held-out anchor against the calibration an unlabelled frame would get.

    Every anchor except the first is held out in turn: it is removed, and the gap it
    leaves is spanned by cross-fading propagation forward from the anchor before it with
    propagation back from the anchor after -- what the calibrator does everywhere else.
    The last anchor has no successor, so it falls back to one-way propagation and is
    scored on that rather than dropped from the population.

    ``frame_shape`` is the ``(height, width)`` the stream was calibrated in, and is what
    enables the calibrator's plausibility gate on each anchor fit. Passing it makes this
    accept exactly the anchors the calibrator accepts; without it the gate is skipped and
    an anchor the stream rejected can seed a score the stream never had.

    Returns ``(px_errors, m_errors, pitch_x, per_anchor_records)``. ``pitch_x`` is the
    labelled pitch x of each measured point, parallel to the error arrays, so
    :func:`region_breakdown` attributes each error to where the point actually is rather
    than to wherever a bad homography projected it.
    """
    frames = sorted(anchors)
    px_all, m_all, x_all, records = [], [], [], []
    for i, held in enumerate(frames[1:], start=1):
        prev = frames[i - 1]
        nxt = frames[i + 1] if i + 1 < len(frames) else None
        span = range(prev, (nxt if nxt is not None else held) + 1)
        missing = [f for f in span if f not in gmc_by_frame]
        if missing:
            raise CalibrationUnmeasurable(
                _unmeasurable_reason(missing[0], prev, max(span), gmc_by_frame)
            )
        neighbours = {f: anchors[f] for f in (prev, nxt) if f is not None}
        fused = fuse_anchors(
            neighbours,
            motions={
                f: FrameMotion(affine=gmc_by_frame[f])
                for f in span
                if f != prev  # motion *into* prev belongs to the previous gap
            },
            frames=list(span),
            spec=spec,
            frame_shape=frame_shape,
            ransac_thresh_px=ransac_thresh_px,
        )
        H = fused[held]
        if H is None:
            raise CalibrationUnmeasurable(
                f"no anchor neighbouring frame {held} produced a usable homography: its "
                "labels could not be solved, or the fit was rejected as implausible"
            )
        img_h, pitch_h = anchors[held]
        px = np.linalg.norm(project(np.linalg.inv(H), pitch_h) - img_h, axis=1)
        m = np.linalg.norm(project(H, img_h) - pitch_h, axis=1)
        px_all.extend(px)
        m_all.extend(m)
        x_all.extend(pitch_h[:, 0])
        records.append(
            {
                "held_frame": held,
                "seeded_from": prev,
                "fused_with": nxt,
                "gap_frames": held - prev,
                "points": int(len(m)),
                "median_m": float(np.median(m)),
                "median_px": float(np.median(px)),
            }
        )
    return np.array(px_all), np.array(m_all), np.array(x_all), records


def check_gate(median_m: float, gate_median_m: float | None) -> list[str]:
    """Return gate failures; empty means pass. ``None`` skips the gate entirely.

    A non-finite median fails rather than passing: ``nan > 0.5`` is False in IEEE
    arithmetic, so a silent pass is the natural bug here.
    """
    if gate_median_m is None:
        return []
    if not np.isfinite(median_m):
        return [f"held-out median is {median_m}, which cannot be compared to a gate"]
    if median_m > gate_median_m:
        return [f"held-out median {median_m:.3f} m exceeds the gate of {gate_median_m} m"]
    return []


def region_breakdown(
    errors_m: np.ndarray,
    pitch_x_m: np.ndarray,
    pitch_length_m: float,
) -> dict[str, dict]:
    """Split held-out errors into pitch thirds along x.

    An oblique camera means error grows with distance, so a single median hides
    which end is wrong. Reported, never gated: this is what tells someone facing a failure
    where to re-click.

    ``pitch_x_m`` is the **labelled** pitch x of each measured point, not the
    projected one: a badly-projected point must be attributed to where it really
    is, not to wherever the bad homography sent it.
    """
    edges = (pitch_length_m / 3.0, 2.0 * pitch_length_m / 3.0)
    masks = {
        "near_third": pitch_x_m < edges[0],
        "middle_third": (pitch_x_m >= edges[0]) & (pitch_x_m < edges[1]),
        "far_third": pitch_x_m >= edges[1],
    }
    out: dict[str, dict] = {}
    for name, mask in masks.items():
        sel = errors_m[mask]
        out[name] = {
            "n": int(sel.size),
            "median_m": float(np.median(sel)) if sel.size else float("nan"),
            "p90_m": float(np.percentile(sel, 90)) if sel.size else float("nan"),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Leave-one-anchor-out accuracy for manual calibration.")
    p.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="calibration.npz written by the calibrate stage.",
    )
    p.add_argument(
        "--anchors",
        type=Path,
        required=True,
        help="The hand-labelled pitch_points.json the calibration was built from.",
    )
    p.add_argument(
        "--ransac-thresh-px",
        type=float,
        default=RANSAC_THRESH_PX,
        help="Anchor-fit RANSAC threshold, in image pixels. Defaults to the value the "
        "calibrator itself fits anchors with.",
    )
    p.add_argument("--output", type=Path, default=None, help="Optional metrics JSON out.")
    p.add_argument(
        "--gate-median-m",
        type=float,
        default=None,
        help="Exit non-zero if the held-out median exceeds this, in metres; the "
        "pipeline gates at 0.5. Omit to report without gating.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Clip config, read for pitch.length / pitch.width. Omit to assume the standard.",
    )
    args = p.parse_args()

    # The fused scorer cross-fades through control points derived from the pitch, so a
    # non-standard venue changes the number this harness reports.
    spec = Config.from_yaml(args.config).pitch.to_spec() if args.config else PitchSpec()

    with np.load(args.calibration) as data:
        gmc_by_frame = {
            int(idx): gmc for idx, gmc in zip(data["frame_idx"], data["gmc"], strict=True)
        }
    frame_shape = load_frame_shape(args.calibration)
    anchors = load_anchors(args.anchors)

    # Every note below records something that changes what this number means. A metric
    # that quietly redefines itself between runs is the failure mode this project treats
    # as worst, so each goes to stderr, and each count into the JSON.
    if frame_shape is None:
        print(
            "note: this calibration.npz records no frame shape, so anchor fits cannot be "
            "checked against the frame bounds the calibrator applies -- anchors it would "
            "have rejected may be scored here. Re-run calibration to record the shape.",
            file=sys.stderr,
        )

    # `load_anchors` drops phantom and individually-excluded clicks by default, so count
    # under each setting to report what this run actually scored.
    def _count(**kwargs: bool) -> int:
        return sum(len(v[0]) for v in load_anchors(args.anchors, **kwargs).values())

    n_all = _count(drop_phantom=False, drop_excluded=False)
    n_after_phantom = _count(drop_excluded=False)  # phantoms gone, marked clicks still in
    n_kept = _count()  # both exclusions applied — what is actually scored
    n_phantom = n_all - n_after_phantom
    n_excluded = n_after_phantom - n_kept
    if n_phantom:
        print(
            f"note: excluded {n_phantom} of {n_all} labelled clicks on phantom vertices "
            f"(unlocalisable points on the penalty-box line; see "
            f"homography.pitch.PHANTOM_VERTICES). "
            f"Numbers are NOT comparable with runs made before this exclusion.",
            file=sys.stderr,
        )
    if n_excluded:
        print(
            f"note: excluded a further {n_excluded} individually-marked clicks "
            f'(per-point "exclude" reasons in {args.anchors.name}). '
            f"Numbers are NOT comparable with runs made before those marks.",
            file=sys.stderr,
        )

    try:
        _check_provenance(args.calibration, args.anchors)
        px, m, pitch_x, records = holdout_errors(
            anchors,
            gmc_by_frame,
            args.ransac_thresh_px,
            spec=spec,
            frame_shape=frame_shape,
        )
    except CalibrationUnmeasurable as exc:
        # No measurement means no evidence the calibration is sound, which is exactly what
        # the gate exists to stop on. Report it the way a failed threshold is reported --
        # a traceback names the line that noticed, not the setting that caused it.
        print(f"GATE FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    metrics = {
        "held_out_anchors": len(records),
        "points": int(len(m)),
        "phantom_clicks_excluded": n_phantom,
        "marked_clicks_excluded": n_excluded,
        "median_m": float(np.median(m)),
        "p90_m": float(np.percentile(m, 90)),
        "median_px": float(np.median(px)),
        "p90_px": float(np.percentile(px, 90)),
        "by_region": region_breakdown(m, pitch_x, spec.length),
        "per_anchor": records,
    }
    print(json.dumps(metrics, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2))

    problems = check_gate(metrics["median_m"], args.gate_median_m)
    if problems:
        for problem in problems:
            print(f"GATE FAIL: {problem}", file=sys.stderr)
        raise SystemExit(1)
    if args.gate_median_m is not None:
        print(f"GATE PASS: median {metrics['median_m']:.3f} m <= {args.gate_median_m} m")
