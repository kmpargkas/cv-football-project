"""The end-to-end stage graph: what each stage needs and what it produces.

Skipping is by output presence, so a rerun after a crash resumes rather than
repeating hours of detection.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Stage:
    """One pipeline step: what it needs and what it makes.

    ``skip_ok=False`` marks a stage whose outputs on disk are not proof it passed;
    the planner always reruns it.
    """

    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    skip_ok: bool = True


# Filenames are relative to the clip's output directory. ``external:`` inputs come
# from outside it (the video, the anchor labels, the weights).
_BASE_STAGES: tuple[Stage, ...] = (
    Stage(
        name="calibrate",
        inputs=("external:video", "external:anchors"),
        # Skip key is deliberately the two files downstream stages depend on, not
        # calibration.mp4 -- calibration_metrics.json is written last, so a crash
        # mid-video-render can't fool the planner into skipping.
        outputs=("calibration.npz", "calibration_metrics.json"),
    ),
    Stage(
        name="gate",
        inputs=("calibration.npz", "external:anchors"),
        outputs=("holdout_metrics.json",),
        # See Stage.skip_ok's docstring: holdout_metrics.json exists on disk even when
        # the gate failed, so its presence can never be treated as "this passed."
        skip_ok=False,
    ),
    Stage(
        name="players",
        inputs=("external:video", "calibration.npz", "external:weights"),
        outputs=("tracks.txt", "teams.csv", "palette.json"),
    ),
    Stage(
        name="stitch",
        inputs=("tracks.txt", "teams.csv"),
        outputs=("stitched.txt",),
    ),
    Stage(
        name="ball",
        inputs=("external:video", "calibration.npz", "external:ball_weights"),
        outputs=("ball.csv", "candidates.csv"),
    ),
    Stage(
        name="project",
        inputs=("stitched.txt", "teams.csv", "ball.csv", "calibration.npz"),
        outputs=("positions.csv", "ball_positions.csv", "possession.csv"),
    ),
    Stage(
        name="radar",
        inputs=("positions.csv", "ball_positions.csv", "possession.csv", "teams.csv"),
        outputs=("radar.mp4",),
    ),
    Stage(
        name="annotate",
        inputs=(
            "external:video",
            "stitched.txt",
            "teams.csv",
            "palette.json",
            "ball.csv",
            "positions.csv",
            "ball_positions.csv",
            "possession.csv",
        ),
        outputs=("demo.mp4",),
    ),
)


def build_stages(ball_weights_required: bool = True) -> tuple[Stage, ...]:
    """The stage graph, minus the edge one run-time choice removes.

    ``ball_weights_required=False``: ``ball.weights`` unset is a supported configuration
    (the ROI pass reuses the player detector), so the ``ball`` stage must not require it.
    """
    dropped: set[str] = set()
    if not ball_weights_required:
        dropped.add("external:ball_weights")
    return tuple(
        replace(
            s,
            inputs=tuple(i for i in s.inputs if i not in dropped),
            outputs=tuple(o for o in s.outputs if o not in dropped),
        )
        for s in _BASE_STAGES
    )


STAGES: tuple[Stage, ...] = build_stages()


def missing_inputs(stage: Stage, out_dir: str | Path, supplied: set[str]) -> list[str]:
    """Inputs this stage needs that are neither on disk nor externally supplied."""
    root = Path(out_dir)
    missing = []
    for name in stage.inputs:
        if name in supplied:
            continue
        if name.startswith("external:") or not (root / name).exists():
            missing.append(name)
    return missing


def plan_stages(
    stages: tuple[Stage, ...],
    out_dir: str | Path,
    force_from: str | None = None,
) -> list[Stage]:
    """Stages still needing a run, in order.

    A stage is skipped only when **every** output exists -- a half-written stage
    is not a completed one. ``force_from`` reruns that stage and all after it,
    which is what you want when an upstream input changed.
    """
    root = Path(out_dir)
    names = [s.name for s in stages]

    if force_from is not None:
        if force_from not in names:
            raise ValueError(f"unknown stage {force_from!r}; known: {names}")
        return list(stages[names.index(force_from) :])

    def done(stage: Stage) -> bool:
        if not stage.skip_ok:
            return False
        return bool(stage.outputs) and all((root / o).exists() for o in stage.outputs)

    return [s for s in stages if not done(s)]
