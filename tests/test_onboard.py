"""New-clip onboarding: the step between 'here is an mp4' and 'run the pipeline'.

Anchor spacing is a computed quantity: a 0.5 m gate failed at 198-frame gaps and
passed at ~99.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from football_tracker.config import Config
from football_tracker.pipeline.onboard import (
    onboarding_checklist,
    scaffold_config,
    suggest_anchor_frames,
)


class TestSuggestAnchorFrames:
    def test_anchors_start_at_zero_and_end_at_the_last_frame(self):
        frames = suggest_anchor_frames(1390, max_gap=100)
        assert frames[0] == 0
        assert frames[-1] == 1389

    @pytest.mark.parametrize("count", [200, 732, 1033, 1390, 5000])
    def test_no_gap_exceeds_the_budget(self, count: int):
        frames = suggest_anchor_frames(count, max_gap=100)
        gaps = [b - a for a, b in zip(frames, frames[1:], strict=False)]
        assert max(gaps) <= 100, f"{count} frames produced a {max(gaps)}-frame gap"

    def test_a_1390_frame_clip_gets_about_fifteen_anchors(self):
        """8 anchors over 1390 frames failed the 0.5 m gate at 0.937 m; ~15 passed."""
        assert 14 <= len(suggest_anchor_frames(1390, max_gap=100)) <= 16

    def test_a_short_clip_still_gets_two_anchors(self):
        frames = suggest_anchor_frames(30, max_gap=100)
        assert len(frames) >= 2
        assert frames[0] == 0
        assert frames[-1] == 29

    def test_a_one_frame_clip_raises_rather_than_returning_a_useless_plan(self):
        with pytest.raises(ValueError, match="at least 2"):
            suggest_anchor_frames(1, max_gap=100)

    def test_anchors_are_sorted_and_unique(self):
        frames = suggest_anchor_frames(1033, max_gap=100)
        assert frames == sorted(set(frames))


class TestScaffoldConfig:
    def test_it_carries_the_clip_into_every_path(self):
        yaml_text = scaffold_config("data/raw/newmatch.mp4", fps=50.0)
        assert "newmatch.mp4" in yaml_text
        assert "outputs/newmatch" in yaml_text

    def test_it_carries_the_clip_fps_not_the_default(self):
        """projection.fps drives ball speed, the unknown window, and the writer rate."""
        assert "fps: 25.0" in scaffold_config("a.mp4", fps=25.0)

    def test_the_scaffolded_config_loads(self, tmp_path: Path):
        """The optional match block ships commented out, so `match` is None."""
        p = tmp_path / "newmatch.yaml"
        p.write_text(scaffold_config("data/raw/newmatch.mp4", fps=50.0))
        cfg = Config.from_yaml(p)
        assert cfg.match is None
        assert cfg.projection.fps == 50.0
        assert cfg.io.output_dir == Path("outputs/newmatch")
        assert cfg.tracking.calibration_path == Path("outputs/newmatch/calibration.npz")

    def test_it_declares_its_pitch(self, tmp_path: Path):
        """Wrong pitch dimensions never raise, so the field must at least be visible."""
        p = tmp_path / "newmatch.yaml"
        p.write_text(scaffold_config("data/raw/newmatch.mp4", fps=50.0))
        cfg = Config.from_yaml(p)
        assert (cfg.pitch.length, cfg.pitch.width) == (105.0, 68.0)


class TestOnboardingChecklist:
    def test_it_reports_what_is_missing(self, tmp_path: Path):
        missing = onboarding_checklist(tmp_path / "gone.json", tmp_path / "no.yaml")
        assert any("anchors" in m for m in missing)
        assert any("config" in m for m in missing)

    def test_it_is_empty_when_everything_is_present(self, tmp_path: Path):
        anchors = tmp_path / "a.json"
        config = tmp_path / "c.yaml"
        anchors.write_text("{}")
        config.write_text("io: {}")
        assert onboarding_checklist(anchors, config) == []


def test_the_annotator_default_layout_matches_the_planner():
    """Without --frames the annotator must derive its layout from the clip length,
    never a fixed count."""
    source = Path(__file__).parent.parent / "scripts" / "annotate_pitch_points.py"
    text = source.read_text()
    assert "suggest_anchor_frames(total)" in text
    assert "default=8" not in text.replace(" ", "")
