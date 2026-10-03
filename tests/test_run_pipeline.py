"""Parity between the stage graph (stages.py) and the runner's command table."""

from __future__ import annotations

from scripts.run_pipeline import STAGE_COMMANDS, RunContext

from football_tracker.pipeline.stages import STAGES


def _ctx(ball_weights: str | None) -> RunContext:
    return RunContext(
        config="cfg.yaml",
        out_dir="out",
        stem="clip",
        video="v.mp4",
        ball_weights=ball_weights,
    )


def test_stage_commands_covers_every_declared_stage():
    declared = {s.name for s in STAGES}
    covered = set(STAGE_COMMANDS)
    assert declared == covered


def test_ball_stage_drives_both_passes_with_the_ball_detector():
    """``ball.weights`` reaches only the ROI pass; without ``--weights`` the full-frame
    candidates still come from the player detector."""
    command = STAGE_COMMANDS["ball"](_ctx("weights/ball.pt"))
    assert "--weights" in command
    assert command[command.index("--weights") + 1] == "weights/ball.pt"


def test_ball_stage_omits_weights_when_no_ball_detector_is_configured():
    """Emitting ``--weights None`` would turn a supported configuration into a
    file-not-found at stage start."""
    command = STAGE_COMMANDS["ball"](_ctx(None))
    assert "--weights" not in command


def test_radar_stage_hands_the_fitted_palette_to_the_renderer():
    command = STAGE_COMMANDS["radar"](_ctx(None))
    assert command[command.index("--palette") + 1] == "out/palette.json"


def test_annotate_stage_renders_the_deliverable_from_the_fitted_palette():
    command = STAGE_COMMANDS["annotate"](_ctx(None))
    assert command[command.index("--palette") + 1] == "out/palette.json"
    assert command[command.index("--positions") + 1] == "out/positions.csv"
    assert command[command.index("--ball-positions") + 1] == "out/ball_positions.csv"
    assert command[command.index("--output") + 1] == "out/demo.mp4"


def test_the_pipeline_ends_with_the_deliverable():
    assert STAGES[-1].name == "annotate"
    assert STAGES[-1].outputs == ("demo.mp4",)
    assert "composite" not in {s.name for s in STAGES}


def test_annotate_stage_draws_possession_from_the_project_stage():
    command = STAGE_COMMANDS["annotate"](_ctx(None))
    assert command[command.index("--possession") + 1] == "out/possession.csv"
    annotate = next(s for s in STAGES if s.name == "annotate")
    assert "possession.csv" in annotate.inputs
