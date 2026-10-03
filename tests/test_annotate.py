"""The tracking annotators, over synthetic frames."""

from __future__ import annotations

import numpy as np

from football_tracker.reid.roles import Role
from football_tracker.reid.teamid import NEUTRAL_BGR, KitColours, TrackLabel
from football_tracker.tracking.annotate import (
    BallAnnotator,
    KitAnnotator,
    TrackAnnotator,
    draw_ball_marker,
)
from football_tracker.tracking.ball_trajectory import BallState
from football_tracker.tracking.tracker import TrackedObjects


def _state(source: str, airborne: bool = False, xy=(128.0, 128.0)) -> BallState:
    return BallState(
        frame_idx=0,
        xy=None if source == "missing" else np.array(xy),
        source=source,
        airborne=airborne,
        confidence=0.9 if source == "detected" else 0.0,
    )


class TestBallAnnotator:
    def test_draws_for_each_positional_source(self):
        for source in ("detected", "interpolated"):
            frame = np.zeros((256, 256, 3), dtype=np.uint8)
            out = BallAnnotator().annotate(frame, _state(source))
            assert out.any(), source
            assert not frame.any()  # input never mutated

    def test_missing_with_no_trail_returns_unchanged_copy(self):
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        out = BallAnnotator().annotate(frame, _state("missing"))
        assert not out.any()
        assert out is not frame

    def test_airborne_marker_differs_from_ground(self):
        ground = BallAnnotator().annotate(
            np.zeros((256, 256, 3), dtype=np.uint8), _state("detected")
        )
        air = BallAnnotator().annotate(
            np.zeros((256, 256, 3), dtype=np.uint8), _state("detected", airborne=True)
        )
        assert not np.array_equal(ground, air)

    def test_trail_lingers_then_fades_out_while_the_ball_is_missing(self):
        """A deque only evicts on append; without an explicit pop the trail would stay
        burned onto every frame of a missing stretch."""
        blank = lambda: np.zeros((256, 256, 3), dtype=np.uint8)  # noqa: E731
        annotator = BallAnnotator(trail_length=8)
        for _ in range(8):
            annotator.annotate(blank(), _state("detected"))

        assert annotator.annotate(blank(), _state("missing")).any(), "should still linger"
        for _ in range(6):
            annotator.annotate(blank(), _state("missing"))
        assert not annotator.annotate(blank(), _state("missing")).any(), "should have faded out"

    def test_prediction_cross_drawn_when_given(self):
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        out = BallAnnotator().annotate(frame, _state("missing"), prediction=np.array([64.0, 64.0]))
        assert out.any()


# --- role colouring ------------------------------------------------------------------


def _team_tracks():
    return TrackedObjects(
        xyxy=np.array([[10.0, 10.0, 40.0, 80.0], [60.0, 10.0, 90.0, 80.0]], dtype=np.float32),
        id=np.array([3, 18], dtype=np.int64),
        confidence=np.array([1.0, 1.0], dtype=np.float32),
        class_id=np.array([2, 2], dtype=np.int64),
    )


def test_team_colouring_distinguishes_a_referee_from_a_team():
    """A referee must never be drawn in a team colour."""
    teams = {
        3: TrackLabel(Role.TEAM_A, 0, 1.0, 100, 0),
        18: TrackLabel(Role.REFEREE, -1, 0.95, 100, 0),
    }
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    out = TrackAnnotator(teams=teams).annotate(frame, _team_tracks())
    assert out.shape == frame.shape
    assert out.any()  # something was drawn
    assert not frame.any()  # input never mutated


def test_a_track_missing_from_teams_csv_renders_as_unknown():
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    out = TrackAnnotator(teams={}).annotate(frame, _team_tracks())
    assert out.shape == frame.shape


def test_without_teams_the_annotator_keeps_per_id_colouring():
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    out = TrackAnnotator().annotate(frame, _team_tracks())
    assert out.shape == frame.shape
    assert out.any()


def test_team_colouring_handles_an_empty_frame_of_tracks():
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    out = TrackAnnotator(teams={}).annotate(frame, TrackedObjects.empty())
    assert not out.any()


# --- deliverable: kit-coloured ellipses, no boxes, no text ----------------------------


def _colours() -> KitColours:
    return KitColours(
        team=((0, 0, 255), (255, 255, 255)), goalkeepers=((0, 255, 255),), referee=(0, 0, 0)
    )


def test_kit_annotator_draws_each_track_in_its_own_kit_colour():
    teams = {
        3: TrackLabel(Role.TEAM_A, 0, 1.0, 100, 0),
        18: TrackLabel(Role.GOALKEEPER, 0, 1.0, 100, 0),
    }
    frame = np.full((100, 120, 3), 90, dtype=np.uint8)
    out = KitAnnotator(teams, _colours()).annotate(frame, _team_tracks())
    assert (frame == 90).all()  # input never mutated
    # Each arc's lowest point sits under the box's bottom centre (y2=80 + minor axis 5).
    assert (out[83:88, 23:28] == (0, 0, 255)).all(axis=2).any()
    assert (out[83:88, 73:78] == (0, 255, 255)).all(axis=2).any()


def test_kit_annotator_draws_nothing_above_the_head_or_between_players():
    """No box, no label: the glow and arc stay within each player's own footprint."""
    frame = np.full((100, 120, 3), 90, dtype=np.uint8)
    out = KitAnnotator({}, _colours()).annotate(frame, _team_tracks())
    assert (out[:4] == 90).all()  # above both boxes (y1=10)
    assert np.abs(out[20:70, 50].astype(int) - 90).max() <= 1  # the gap between the boxes
    assert (out[73:88] != 90).any()


def test_kit_annotator_glow_pools_at_the_feet_and_fades_up_the_body():
    teams = {3: TrackLabel(Role.TEAM_A, 0, 1.0, 100, 0)}  # red (0, 0, 255)
    frame = np.full((100, 120, 3), 90, dtype=np.uint8)
    out = KitAnnotator(teams, _colours()).annotate(frame, _team_tracks())
    feet, body, head = out[79, 25].astype(int), out[60, 25].astype(int), out[14, 25].astype(int)
    assert feet[2] > 150 and feet[0] < 60  # strongly red-shifted inside the pool
    assert 100 < body[2] < feet[2]  # lit, but weaker than the feet
    assert np.abs(head - 90).max() <= 1  # the wash has faded out by the head


def test_kit_annotator_uses_neutral_for_an_unlabelled_track():
    frame = np.full((100, 120, 3), 90, dtype=np.uint8)
    out = KitAnnotator({}, _colours()).annotate(frame, _team_tracks())
    assert (out[83:88, 23:28] == NEUTRAL_BGR).all(axis=2).any()


def test_kit_annotator_handles_an_empty_frame_of_tracks():
    frame = np.full((100, 120, 3), 90, dtype=np.uint8)
    out = KitAnnotator({}, _colours()).annotate(frame, TrackedObjects.empty())
    assert (out == 90).all()


def test_kit_annotator_id_pill_sits_under_the_arc_in_the_kit_colour():
    teams = {3: TrackLabel(Role.TEAM_A, 0, 1.0, 100, 0)}  # red (0, 0, 255)
    frame = np.full((140, 120, 3), 90, dtype=np.uint8)
    out = KitAnnotator(teams, _colours(), labels={3: "7"}).annotate(frame, _team_tracks())
    below_arc = out[88:115, 10:40]
    assert (below_arc == (0, 0, 255)).all(axis=2).any()  # pill fill
    assert (below_arc.max(axis=2) > 200).any() and (below_arc.min(axis=2) < 60).any()
    # track 18 has no label: its footprint is identical to the label-free render
    plain = KitAnnotator(teams, _colours()).annotate(frame, _team_tracks())
    np.testing.assert_array_equal(out[:, 50:], plain[:, 50:])


def test_kit_annotator_id_text_is_dark_on_a_light_kit():
    teams = {18: TrackLabel(Role.TEAM_B, 1, 1.0, 100, 0)}  # white (255, 255, 255)
    frame = np.full((140, 120, 3), 90, dtype=np.uint8)
    out = KitAnnotator(teams, _colours(), labels={18: "10"}).annotate(frame, _team_tracks())
    pill = out[88:115, 55:95]
    assert (pill == (255, 255, 255)).all(axis=2).any()
    assert (pill.max(axis=2) < 60).any()  # dark digits inside the white pill


def test_kit_annotator_without_labels_draws_no_text():
    frame = np.full((140, 120, 3), 90, dtype=np.uint8)
    plain = KitAnnotator({}, _colours()).annotate(frame, _team_tracks())
    unlabelled = KitAnnotator({}, _colours(), labels={}).annotate(frame, _team_tracks())
    np.testing.assert_array_equal(plain, unlabelled)
    assert (plain[110:] == 90).all()  # below the glow's reach, nothing at all


def test_ball_marker_is_a_black_ring_that_leaves_the_ball_visible():
    frame = np.full((64, 64, 3), 200, dtype=np.uint8)
    out = draw_ball_marker(frame, np.array([32.0, 32.0]), radius=10)
    assert (frame == 200).all()
    assert tuple(out[32, 32]) == (200, 200, 200)  # hollow
    assert (out[32, 20:25] == 0).all(axis=1).any()  # ring on the left edge
