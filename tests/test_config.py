"""The YAML-backed configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from football_tracker.config import Config
from football_tracker.pipeline.onboard import scaffold_config
from football_tracker.utils import resolve_device


class TestConfig:
    def test_defaults(self):
        cfg = Config()
        assert cfg.device == "auto"
        assert cfg.io.auto_letterbox is True
        assert cfg.match is None  # optional display names, never defaulted

    def test_yaml_round_trip(self, tmp_path: Path):
        cfg = Config(device="cpu")
        cfg.io.input_video = Path("data/raw/clip.mp4")
        cfg.io.frame_stride = 3

        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))

        assert Config.from_yaml(path) == cfg

    def test_partial_yaml_falls_back_to_defaults(self, tmp_path: Path):
        path = tmp_path / "partial.yaml"
        path.write_text("device: cpu\n")

        cfg = Config.from_yaml(path)
        assert cfg.device == "cpu"
        assert cfg.io.auto_letterbox is True  # untouched default
        assert cfg.io.frame_stride == 1

    def test_empty_yaml_is_all_defaults(self, tmp_path: Path):
        path = tmp_path / "empty.yaml"
        path.write_text("")
        assert Config.from_yaml(path) == Config()


class TestBallConfig:
    def test_defaults(self):
        cfg = Config()
        assert cfg.ball.enabled is True
        assert cfg.ball.roi_size == 640
        assert cfg.ball.roi_conf == pytest.approx(0.10)
        assert cfg.ball.mask_margin_m == pytest.approx(6.0)
        assert cfg.ball.max_gap == 60
        assert cfg.ball.max_interp_gap == 30

    def test_yaml_ball_section_parses(self, tmp_path: Path):
        path = tmp_path / "ball.yaml"
        path.write_text("ball:\n  roi_size: 512\n  miss_cost: 0.5\n")
        cfg = Config.from_yaml(path)
        assert cfg.ball.roi_size == 512
        assert cfg.ball.miss_cost == pytest.approx(0.5)
        assert cfg.ball.roi_conf == pytest.approx(0.10)  # untouched default

    def test_yaml_round_trip_includes_ball(self, tmp_path: Path):
        cfg = Config()
        cfg.ball.max_speed_px = 55.0
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
        assert Config.from_yaml(path) == cfg


class TestResolveDevice:
    @pytest.mark.parametrize("preference", ["cpu", "mps", "cuda"])
    def test_explicit_preference_is_returned_unchanged(self, preference: str):
        assert resolve_device(preference) == preference

    def test_auto_resolves_to_a_real_device(self):
        assert resolve_device("auto") in {"cpu", "mps", "cuda"}


def test_projection_config_defaults():
    cfg = Config()
    assert cfg.projection.smooth_window == 15
    assert cfg.projection.fps == 60.0
    assert cfg.projection.possession.radius_m == 2.0
    assert cfg.projection.possession.acquire_frames == 12
    assert cfg.match is None


def test_match_config_from_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("match:\n  team_a_name: Home (red)\n  team_b_name: Away (white)\n")
    cfg = Config.from_yaml(p)
    assert cfg.match is not None
    assert cfg.match.team_b_name == "Away (white)"


def _generated(tmp_path: Path, stem: str = "newmatch") -> Path:
    """A config as `scripts/new_clip.py` writes it, with the `match:` block filled in."""
    text = scaffold_config(f"data/raw/{stem}.mp4", fps=60.0)
    text += 'match:\n  team_a_name: "A"\n  team_b_name: "B"\n'
    path = tmp_path / f"{stem}.yaml"
    path.write_text(text)
    return path


def test_generated_config_is_self_contained(tmp_path: Path):
    """A generated config names its own video, calibration cache, and match metadata;
    nothing a run depends on may live outside it."""
    cfg = Config.from_yaml(_generated(tmp_path))

    assert cfg.io.input_video is not None
    assert cfg.io.input_video.stem == "newmatch"
    assert "newmatch" in str(cfg.tracking.calibration_path)
    assert cfg.io.frame_stride == 1  # projection assumes consecutive indices
    assert cfg.detection.agnostic_nms is True  # the phase5k result requires it
    assert cfg.match is not None


def test_generated_config_output_dir_is_clip_scoped(tmp_path: Path):
    """Two clips must never write into one directory."""
    cfg = Config.from_yaml(_generated(tmp_path))
    assert "newmatch" in str(cfg.io.output_dir)


def test_pitch_config_defaults_to_fifa_standard():
    cfg = Config()
    assert cfg.pitch.length == 105.0
    assert cfg.pitch.width == 68.0


def test_pitch_config_to_spec_round_trips(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text("pitch:\n  length: 100.0\n  width: 64.0\n")
    spec = Config.from_yaml(p).pitch.to_spec()
    assert spec.length == 100.0
    assert spec.width == 64.0


def test_pitch_config_rejects_dimensions_outside_the_laws_of_the_game(tmp_path: Path):
    """FIFA: 100-110 m x 64-75 m. A typo here silently rescales every metric."""
    p = tmp_path / "c.yaml"
    p.write_text("pitch:\n  length: 10.0\n  width: 68.0\n")
    with pytest.raises(ValidationError, match="length"):
        Config.from_yaml(p)


def test_pitch_config_checks_width_against_the_laws_too(tmp_path: Path):
    """Width rescales every lateral distance exactly as length rescales longitudinal."""
    p = tmp_path / "c.yaml"
    p.write_text("pitch:\n  length: 105.0\n  width: 6.8\n")
    with pytest.raises(ValidationError, match="width"):
        Config.from_yaml(p)


def test_generated_config_declares_its_pitch(tmp_path: Path):
    """A dimension left implicit is one nobody checked against the venue."""
    assert "pitch:" in _generated(tmp_path).read_text()
