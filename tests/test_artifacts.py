"""The tracking-pass artifact formats: both passes key on the true decode frame index,
and the MOT ±1 shift round-trips exactly."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from football_tracker.tracking.artifacts import (
    mot_row,
    read_ball_csv,
    read_candidates_csv,
    read_mot,
    write_ball_csv,
    write_candidates_csv,
)
from football_tracker.tracking.ball import BallCandidate
from football_tracker.tracking.ball_trajectory import BallState


def _state(frame_idx, xy, source="detected", airborne=False, confidence=0.9) -> BallState:
    return BallState(
        frame_idx=frame_idx,
        xy=None if xy is None else np.array(xy, dtype=np.float64),
        source=source,
        airborne=airborne,
        confidence=confidence,
    )


class TestBallCsv:
    def test_round_trip_preserves_every_field(self, tmp_path: Path):
        states = [
            _state(100, (12.25, 34.5), "detected", False, 0.87),
            _state(101, (13.0, 35.0), "interpolated", True, 0.0),
            _state(102, None, "missing", False, 0.0),
        ]
        path = tmp_path / "ball.csv"
        write_ball_csv(path, states)
        back = read_ball_csv(path)

        assert set(back) == {100, 101, 102}
        assert back[100].xy == pytest.approx([12.25, 34.5])
        assert back[100].source == "detected" and back[100].confidence == pytest.approx(0.87)
        assert back[101].airborne is True
        assert back[102].xy is None and back[102].source == "missing"

    def test_missing_frames_are_written_not_omitted(self, tmp_path: Path):
        """The CSV is a complete record of the range — gaps are explicit rows."""
        path = tmp_path / "ball.csv"
        write_ball_csv(path, [_state(5, None, "missing", confidence=0.0)])
        assert path.read_text().splitlines()[1].startswith("5,,,missing")

    def test_frame_index_is_the_decode_index_not_renumbered(self, tmp_path: Path):
        """Rows keep true decode indices so passes over the same range join up."""
        path = tmp_path / "ball.csv"
        write_ball_csv(path, [_state(507, (1.0, 2.0)), _state(1389, (3.0, 4.0))])
        assert sorted(read_ball_csv(path)) == [507, 1389]


class TestMot:
    def test_row_shifts_top_left_by_one_and_keeps_wh(self):
        row = mot_row(7, 3, 2, np.array([10.0, 20.0, 30.0, 60.0]))
        frame, tid, x, y, w, h, conf, cls, vis = row.split(",")
        assert (frame, tid, cls) == ("7", "3", "2")
        assert (float(x), float(y)) == (11.0, 21.0)  # MOT is 1-indexed
        assert (float(w), float(h)) == (20.0, 40.0)  # unshifted

    def test_round_trip_recovers_the_original_box(self, tmp_path: Path):
        xyxy = np.array([10.0, 20.0, 30.0, 60.0])
        path = tmp_path / "tracks.txt"
        path.write_text(mot_row(7, 3, 2, xyxy) + "\n")
        tracks = read_mot(path)[7]
        assert tracks.xyxy[0] == pytest.approx(xyxy)
        assert int(tracks.id[0]) == 3
        assert int(tracks.class_id[0]) == 2

    def test_groups_multiple_tracks_per_frame(self, tmp_path: Path):
        path = tmp_path / "tracks.txt"
        path.write_text(
            "\n".join(
                [
                    mot_row(4, 1, 2, np.array([0.0, 0.0, 10.0, 10.0])),
                    mot_row(4, 2, 3, np.array([50.0, 50.0, 60.0, 70.0])),
                    mot_row(5, 1, 2, np.array([1.0, 1.0, 11.0, 11.0])),
                ]
            )
            + "\n"
        )
        by_frame = read_mot(path)
        assert sorted(by_frame) == [4, 5]
        assert len(by_frame[4]) == 2
        assert set(by_frame[4].id.tolist()) == {1, 2}
        assert len(by_frame[5]) == 1

    def test_frame_field_is_not_reindexed(self, tmp_path: Path):
        """MOT coordinates are 1-indexed; the frame column deliberately is not."""
        path = tmp_path / "tracks.txt"
        path.write_text(mot_row(0, 1, 2, np.array([5.0, 5.0, 15.0, 15.0])) + "\n")
        assert 0 in read_mot(path)  # frame 0 survives as frame 0

    def test_empty_file_is_no_tracks(self, tmp_path: Path):
        path = tmp_path / "tracks.txt"
        path.write_text("")
        assert read_mot(path) == {}


class TestCandidatesCsv:
    def test_round_trip(self, tmp_path: Path):
        cands = [
            BallCandidate(
                frame_idx=10,
                xy=np.array([100.0, 200.0], dtype=np.float32),
                wh=np.array([12.0, 12.0], dtype=np.float32),
                confidence=0.8,
                source="full",
            ),
            BallCandidate(
                frame_idx=10,
                xy=np.array([105.0, 202.0], dtype=np.float32),
                wh=np.array([11.0, 11.0], dtype=np.float32),
                confidence=0.3,
                source="roi",
            ),
        ]
        path = tmp_path / "cand.csv"
        write_candidates_csv(path, cands)
        back = read_candidates_csv(path)
        assert len(back[10]) == 2
        assert back[10][0] == (100.0, 200.0, 0.8, "full")
        assert back[10][1] == (105.0, 202.0, 0.3, "roi")

    def test_groups_candidates_by_frame(self, tmp_path: Path):
        path = tmp_path / "cand.csv"
        path.write_text(
            "frame,x,y,confidence,source\n"
            "10,100.00,200.00,0.800,full\n"
            "10,105.00,202.00,0.300,roi\n"
            "11,110.00,205.00,0.900,roi\n"
        )
        cands = read_candidates_csv(path)
        assert len(cands[10]) == 2 and len(cands[11]) == 1
        assert cands[10][0] == (100.0, 200.0, 0.8, "full")
