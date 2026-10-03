"""The pre-flight checks shared by the stage scripts."""

from pathlib import Path

import pytest

from football_tracker.config import Config
from football_tracker.pipeline.cli import (
    check_projection_fps,
    require_contiguous_frames,
    require_file,
)


def test_require_file_returns_the_path_when_it_exists(tmp_path: Path):
    f = tmp_path / "x.csv"
    f.write_text("")
    assert require_file(str(f), "tracks") == f


def test_require_file_exits_naming_the_argument(tmp_path: Path):
    with pytest.raises(SystemExit, match="--tracks not found"):
        require_file(tmp_path / "missing.csv", "tracks")


def test_stride_other_than_one_exits():
    cfg = Config()
    cfg.io.frame_stride = 2
    with pytest.raises(SystemExit, match="frame_stride=2"):
        require_contiguous_frames(cfg)


def test_stride_of_one_passes():
    require_contiguous_frames(Config())


def test_fps_check_is_skipped_without_a_video():
    cfg = Config()
    cfg.io.input_video = None
    check_projection_fps(cfg)


def test_unprobeable_video_only_warns(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    cfg = Config()
    cfg.io.input_video = tmp_path / "missing.mp4"
    check_projection_fps(cfg)
    assert "warning: could not probe" in capsys.readouterr().err


def test_fps_mismatch_exits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import football_tracker.pipeline.cli as cli
    from football_tracker.io.video import VideoMetadata

    cfg = Config()
    cfg.io.input_video = tmp_path / "clip.mp4"
    cfg.projection.fps = 60.0
    monkeypatch.setattr(
        cli,
        "probe_video",
        lambda p: VideoMetadata(path=Path(p), width=2, height=2, fps=30.0, frame_count=1),
    )
    with pytest.raises(SystemExit, match="does not match the probed fps"):
        check_projection_fps(cfg)
