"""Stage planning for the end-to-end pipeline."""

from __future__ import annotations

import pytest

from football_tracker.pipeline.stages import (
    STAGES,
    Stage,
    build_stages,
    missing_inputs,
    plan_stages,
)


def test_stage_order_is_topological():
    """Every stage's inputs are produced by an earlier stage or are external."""
    produced: set[str] = set()
    for stage in STAGES:
        for name in stage.inputs:
            assert name in produced or name.startswith("external:"), (
                f"{stage.name} consumes {name} before anything produces it"
            )
        produced.update(stage.outputs)


def test_plan_runs_everything_when_nothing_exists(tmp_path):
    assert [s.name for s in plan_stages(STAGES, tmp_path)] == [s.name for s in STAGES]


def test_plan_skips_a_stage_whose_outputs_all_exist(tmp_path):
    stages = (
        Stage(name="a", inputs=(), outputs=("a.txt",)),
        Stage(name="b", inputs=("a.txt",), outputs=("b.txt",)),
    )
    (tmp_path / "a.txt").write_text("x")
    assert [s.name for s in plan_stages(stages, tmp_path)] == ["b"]


def test_plan_reruns_a_stage_with_only_some_outputs_present(tmp_path):
    """A half-written stage is not a completed stage."""
    stages = (Stage(name="a", inputs=(), outputs=("a.txt", "a2.txt")),)
    (tmp_path / "a.txt").write_text("x")
    assert [s.name for s in plan_stages(stages, tmp_path)] == ["a"]


def test_force_from_reruns_that_stage_and_everything_after(tmp_path):
    stages = (
        Stage(name="a", inputs=(), outputs=("a.txt",)),
        Stage(name="b", inputs=("a.txt",), outputs=("b.txt",)),
        Stage(name="c", inputs=("b.txt",), outputs=("c.txt",)),
    )
    for name in ("a.txt", "b.txt", "c.txt"):
        (tmp_path / name).write_text("x")
    assert [s.name for s in plan_stages(stages, tmp_path, force_from="b")] == ["b", "c"]


def test_force_from_an_unknown_stage_raises():
    with pytest.raises(ValueError, match="unknown stage"):
        plan_stages(STAGES, "/tmp", force_from="nonsense")


def test_missing_inputs_names_what_is_absent(tmp_path):
    stage = Stage(name="b", inputs=("a.txt", "z.txt"), outputs=("b.txt",))
    (tmp_path / "a.txt").write_text("x")
    assert missing_inputs(stage, tmp_path, supplied=set()) == ["z.txt"]


def test_missing_inputs_treats_external_inputs_as_supplied(tmp_path):
    stage = Stage(name="a", inputs=("external:video",), outputs=("a.txt",))
    assert missing_inputs(stage, tmp_path, supplied={"external:video"}) == []


def test_plan_never_skips_a_stage_with_skip_ok_false(tmp_path):
    """An existing output is not proof of success for a stage marked skip_ok=False."""
    stages = (Stage(name="a", inputs=(), outputs=("a.txt",), skip_ok=False),)
    (tmp_path / "a.txt").write_text("x")
    assert [s.name for s in plan_stages(stages, tmp_path)] == ["a"]


def test_gate_stage_is_never_skipped():
    """scripts/gate_calibration.py writes holdout_metrics.json before evaluating the gate,
    so a failed gate still leaves a "done"-looking file -- gate must always rerun."""
    gate = next(s for s in STAGES if s.name == "gate")
    assert gate.skip_ok is False


def test_build_stages_defaults_to_the_full_graph():
    assert build_stages() == STAGES


def test_ball_weights_required_by_default():
    """Unset ball.weights is a supported fallback, not the default shape of the graph."""
    ball = next(s for s in build_stages() if s.name == "ball")
    assert "external:ball_weights" in ball.inputs


def test_ball_weights_not_required_drops_it_from_balls_inputs():
    """Otherwise a config with ball.weights unset is rejected by the preflight check
    before the command that would have run fine is built."""
    ball = next(s for s in build_stages(ball_weights_required=False) if s.name == "ball")
    assert "external:ball_weights" not in ball.inputs


def test_ball_weights_not_required_graph_is_otherwise_identical():
    base, optional = build_stages(), build_stages(ball_weights_required=False)
    assert [(s.name, s.skip_ok) for s in base] == [(s.name, s.skip_ok) for s in optional]


def test_ball_weights_not_required_graph_is_still_topological():
    produced: set[str] = set()
    for stage in build_stages(ball_weights_required=False):
        for name in stage.inputs:
            assert name in produced or name.startswith("external:")
        produced.update(stage.outputs)


def test_the_fitted_palette_is_a_declared_edge_from_players_to_annotate():
    """The deliverable needs palette.json, so the planner must know who makes it."""
    by_name = {s.name: s for s in STAGES}
    assert "palette.json" in by_name["players"].outputs
    assert "palette.json" in by_name["annotate"].inputs


def test_an_output_dir_without_a_palette_reruns_players(tmp_path):
    for name in ("calibration.npz", "holdout_metrics.json", "tracks.txt", "teams.csv"):
        (tmp_path / name).touch()
    assert "players" in [s.name for s in plan_stages(STAGES, tmp_path)]
