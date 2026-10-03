"""Run the whole pipeline for one clip.

Each stage is a subprocess under `uv run`, executed in graph order, so a stage
that fails stops the run before anything downstream consumes its missing output.

    uv run python scripts/run_pipeline.py --config configs/<clip>.yaml
    uv run python scripts/run_pipeline.py --config configs/<clip>.yaml --from project
    uv run python scripts/run_pipeline.py --config configs/<clip>.yaml --dry-run

Stages whose outputs already exist are skipped, so a rerun after a crash resumes
instead of repeating hours of detection. `--from <stage>` reruns that stage and
everything after it.

Paths are relative to the repo root, so invoke it from there.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from football_tracker.config import Config
from football_tracker.pipeline.stages import (
    STAGES,
    build_stages,
    missing_inputs,
    plan_stages,
)


def _anchors_path(stem: str) -> str:
    """The clip's hand-labelled pitch-anchor JSON, by video stem."""
    return f"data/annotations/pitch_points_{stem}.json"


@dataclass(frozen=True)
class RunContext:
    """Everything a stage command is built from."""

    config: str
    out_dir: str
    stem: str
    video: str
    ball_weights: str | None


STAGE_COMMANDS: dict[str, Callable[[RunContext], list[str]]] = {
    "calibrate": lambda r: [
        "python",
        "scripts/calibrate_video.py",
        "--config",
        r.config,
        "--anchors",
        _anchors_path(r.stem),
        "--output-dir",
        r.out_dir,
    ],
    "gate": lambda r: [
        "python",
        "scripts/gate_calibration.py",
        # The gate is in metres; without --config the holdout assumes a standard pitch.
        "--config",
        r.config,
        "--calibration",
        f"{r.out_dir}/calibration.npz",
        "--anchors",
        _anchors_path(r.stem),
        "--gate-median-m",
        "0.5",
        "--output",
        f"{r.out_dir}/holdout_metrics.json",
    ],
    "players": lambda r: [
        "python",
        "scripts/track_video.py",
        "--config",
        r.config,
        "--no-ball",
        "--calibration",
        f"{r.out_dir}/calibration.npz",
        "--teams",
        "--mot-out",
        f"{r.out_dir}/tracks.txt",
        "--teams-out",
        f"{r.out_dir}/teams.csv",
        "--palette-out",
        f"{r.out_dir}/palette.json",
    ],
    "stitch": lambda r: [
        "python",
        "scripts/stitch_tracks.py",
        "--config",
        r.config,
        "--tracks",
        f"{r.out_dir}/tracks.txt",
        "--teams",
        f"{r.out_dir}/teams.csv",
        "--calibration",
        f"{r.out_dir}/calibration.npz",
        "--video",
        r.video,
        "--out",
        f"{r.out_dir}/stitched.txt",
    ],
    "ball": lambda r: [
        "python",
        "scripts/track_video.py",
        "--config",
        r.config,
        # Seeds both ball passes with the ball detector; `ball.weights` alone reaches
        # only the ROI pass. Unset is supported: the ROI pass then shares the player detector.
        *(["--weights", r.ball_weights] if r.ball_weights else []),
        "--ball",
        "--calibration",
        f"{r.out_dir}/calibration.npz",
        "--ball-out",
        f"{r.out_dir}/ball.csv",
        "--ball-candidates-out",
        f"{r.out_dir}/candidates.csv",
    ],
    "project": lambda r: [
        "python",
        "scripts/project_tracks.py",
        "--config",
        r.config,
        "--tracks",
        f"{r.out_dir}/stitched.txt",
        "--teams",
        f"{r.out_dir}/teams.csv",
        "--ball",
        f"{r.out_dir}/ball.csv",
        "--calibration",
        f"{r.out_dir}/calibration.npz",
        "--out-dir",
        r.out_dir,
    ],
    "radar": lambda r: [
        "python",
        "scripts/render_radar.py",
        "--config",
        r.config,
        "--positions",
        f"{r.out_dir}/positions.csv",
        "--ball",
        f"{r.out_dir}/ball_positions.csv",
        "--possession",
        f"{r.out_dir}/possession.csv",
        "--teams",
        f"{r.out_dir}/teams.csv",
        # Optional on the script side: falls back to placeholder colours when absent.
        "--palette",
        f"{r.out_dir}/palette.json",
        "--out",
        f"{r.out_dir}/radar.mp4",
    ],
    "annotate": lambda r: [
        "python",
        "scripts/render_tracking.py",
        "--config",
        r.config,
        "--tracks",
        f"{r.out_dir}/stitched.txt",
        "--ball",
        f"{r.out_dir}/ball.csv",
        "--teams",
        f"{r.out_dir}/teams.csv",
        "--palette",
        f"{r.out_dir}/palette.json",
        "--positions",
        f"{r.out_dir}/positions.csv",
        "--ball-positions",
        f"{r.out_dir}/ball_positions.csv",
        "--possession",
        f"{r.out_dir}/possession.csv",
        "--output",
        f"{r.out_dir}/demo.mp4",
    ],
}

# A stage in STAGES with no command should fail here, not mid-run after earlier stages
# have burned wall time.
_uncovered = [s.name for s in STAGES if s.name not in STAGE_COMMANDS]
if _uncovered:
    raise RuntimeError(
        f"STAGE_COMMANDS has no entry for stage(s) {_uncovered} declared in STAGES -- "
        "stages.py and run_pipeline.py have drifted apart."
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the end-to-end pipeline for one clip.")
    p.add_argument("--config", required=True, help="Clip config, e.g. configs/<clip>.yaml.")
    p.add_argument("--from", dest="from_stage", default=None, help="Restart at this stage.")
    p.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = Config.from_yaml(args.config)
    out_dir = str(cfg.io.output_dir)
    stages = build_stages(ball_weights_required=cfg.ball.weights is not None)

    # Planned before the preflight check touches disk, so --dry-run always works.
    planned = plan_stages(stages, out_dir, force_from=args.from_stage)
    planned_names = {s.name for s in planned}

    print(f"Plan for {args.config} (out_dir={out_dir}):")
    for stage in stages:
        status = "run" if stage.name in planned_names else "skip"
        print(f"  [{status:4}] {stage.name}")

    if args.dry_run:
        return

    if cfg.io.input_video is None:
        sys.exit("io.input_video is not set in the config.")
    video = str(cfg.io.input_video)
    stem = Path(video).stem
    external_paths = {
        "external:video": Path(video),
        "external:anchors": Path(_anchors_path(stem)),
        "external:weights": Path(cfg.detection.weights) if cfg.detection.weights else None,
        "external:ball_weights": Path(cfg.ball.weights) if cfg.ball.weights else None,
    }
    supplied = {name for name, path in external_paths.items() if path is not None and path.exists()}

    # An input another planned stage produces is not missing, just not made yet; this
    # lets `--from radar` skip checking weights no planned stage opens while still
    # catching `--from radar` when `project` never ran.
    planned_outputs = {o for stage in planned for o in stage.outputs}
    missing: list[str] = []
    for stage in planned:
        for name in missing_inputs(stage, out_dir, supplied):
            if name in planned_outputs or name in missing:
                continue
            missing.append(name)
    if missing:
        print("Missing inputs, refusing to start:")
        for name in missing:
            if name in external_paths:
                path = external_paths[name]
                label = f"{name} ({path})" if path is not None else f"{name} (unset in config)"
            else:
                label = f"{name} (expected in {out_dir}, not produced by any planned stage)"
            print(f"  {label}")
        if any(name in ("external:weights", "external:ball_weights") for name in missing):
            print("Model weights are fetched with: uv run python scripts/pull_weights.py")
        sys.exit(1)

    ctx = RunContext(
        config=args.config,
        out_dir=out_dir,
        stem=stem,
        video=video,
        ball_weights=str(cfg.ball.weights) if cfg.ball.weights else None,
    )
    for stage in planned:
        command = ["uv", "run", *STAGE_COMMANDS[stage.name](ctx)]
        print(f"\n=== {stage.name} ===")
        print(" ".join(command))
        result = subprocess.run(command, check=False)
        if result.returncode < 0:
            # A negative returncode is a signal (e.g. Ctrl-C): report it as an interrupt,
            # not a failure, and exit 128+signal.
            sig = -result.returncode
            print(f"\nStage {stage.name!r} was terminated by signal {sig} (e.g. Ctrl-C).")
            sys.exit(128 + sig)
        if result.returncode != 0:
            # Any non-zero exit stops the run -- a failed gate must not produce a demo.
            print(f"\nStage {stage.name!r} failed (exit {result.returncode}). Re-run with:")
            print(" ".join(command))
            sys.exit(result.returncode)
        if stage.name == "gate":
            print("Gate passed, continuing.")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
