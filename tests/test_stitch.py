"""Offline tracklet stitching: fragments in, one identity per person out."""

from __future__ import annotations

import numpy as np
import pytest

from football_tracker.config import StitchConfig
from football_tracker.reid.roles import Role
from football_tracker.reid.stitch import (
    Tracklet,
    apply_remap,
    build_tracklets,
    link_cost,
    roster_overflow,
    solve_links,
)
from football_tracker.reid.teamid import TrackLabel

FPS = 60.0
TEAM_A, TEAM_B = Role.TEAM_A, Role.TEAM_B


def _label(role: Role, confidence: float = 1.0) -> TrackLabel:
    return TrackLabel(role, -1, confidence, 10, 0)


def _tracklet(track_id, first, last, head, tail, velocity=(0.0, 0.0), role=TEAM_A, n=None):
    return Tracklet(
        track_id=track_id,
        role=role,
        first_frame=first,
        last_frame=last,
        head_xy=np.array(head, dtype=float),
        tail_xy=np.array(tail, dtype=float),
        tail_velocity=np.array(velocity, dtype=float),
        n_frames=n if n is not None else last - first + 1,
    )


class TestBuildTracklets:
    def test_one_tracklet_per_track_with_head_tail_and_velocity(self):
        # A track walking +1 m per frame along x, over 5 frames at 60 fps -> 60 m/s.
        positions = {f: {7: np.array([float(f), 0.0])} for f in range(5)}

        (t,) = build_tracklets(positions, {7: _label(TEAM_A)}, fps=FPS, velocity_window_s=2 / FPS)

        assert (t.track_id, t.first_frame, t.last_frame, t.n_frames) == (7, 0, 4, 5)
        assert t.head_xy == pytest.approx([0.0, 0.0])
        assert t.tail_xy == pytest.approx([4.0, 0.0])
        assert t.tail_velocity[0] == pytest.approx(FPS)  # 1 m per frame

    def test_a_track_with_no_calibrated_position_is_dropped(self):
        # Nothing to reason about geometrically, and geometry is the whole cost model.
        assert build_tracklets({}, {7: _label(TEAM_A)}, fps=FPS) == []

    def test_a_single_frame_tracklet_gets_zero_velocity_not_a_divide_by_zero(self):
        (t,) = build_tracklets({3: {1: np.array([10.0, 10.0])}}, {1: _label(TEAM_A)}, fps=FPS)

        assert t.tail_velocity == pytest.approx([0.0, 0.0])


class TestLinkCost:
    def _cfg(self, **kw):
        return StitchConfig(**kw)

    def test_a_plausible_continuation_is_cheap(self):
        a = _tracklet(1, 0, 100, (0.0, 0.0), (10.0, 0.0), velocity=(5.0, 0.0))
        b = _tracklet(2, 110, 200, (10.8, 0.0), (20.0, 0.0))

        cost = link_cost(a, b, FPS, self._cfg())

        assert cost is not None and cost < 0.5

    def test_temporally_overlapping_tracklets_are_never_linked(self):
        # One person cannot be in two places. This single constraint is the reason the
        # offline pass can do what the online tracker structurally cannot.
        a = _tracklet(1, 0, 100, (0.0, 0.0), (10.0, 0.0))
        b = _tracklet(2, 90, 200, (10.1, 0.0), (20.0, 0.0))

        assert link_cost(a, b, FPS, self._cfg()) is None

    def test_different_teams_are_never_linked(self):
        a = _tracklet(1, 0, 100, (0.0, 0.0), (10.0, 0.0), role=TEAM_A)
        b = _tracklet(2, 110, 200, (10.1, 0.0), (20.0, 0.0), role=TEAM_B)

        assert link_cost(a, b, FPS, self._cfg()) is None

    def test_a_physically_unreachable_gap_is_refused(self):
        # 0.17 s at 9 m/s reaches ~1.5 m; 40 m away is a different person.
        a = _tracklet(1, 0, 100, (0.0, 0.0), (10.0, 0.0))
        b = _tracklet(2, 110, 200, (50.0, 0.0), (60.0, 0.0))

        assert link_cost(a, b, FPS, self._cfg()) is None

    def test_a_gap_longer_than_the_horizon_is_refused_even_if_reachable(self):
        a = _tracklet(1, 0, 100, (0.0, 0.0), (10.0, 0.0))
        b = _tracklet(2, 1000, 1100, (10.1, 0.0), (20.0, 0.0))

        assert link_cost(a, b, FPS, self._cfg(max_gap_s=2.0)) is None

    def test_the_continuation_matching_the_exit_velocity_is_preferred(self):
        # Two candidates equidistant in time; the one where the player kept running in
        # the direction they left in should win. Geometry carries identity here.
        a = _tracklet(1, 0, 100, (0.0, 0.0), (10.0, 0.0), velocity=(6.0, 0.0))
        along = _tracklet(2, 110, 200, (11.0, 0.0), (12.0, 0.0))
        across = _tracklet(3, 110, 200, (10.0, 1.0), (10.0, 2.0))

        cfg = self._cfg()
        assert link_cost(a, along, FPS, cfg) < link_cost(a, across, FPS, cfg)


class TestSolveLinks:
    def test_two_fragments_of_one_person_are_merged(self):
        a = _tracklet(1, 0, 100, (0.0, 0.0), (10.0, 0.0), velocity=(5.0, 0.0))
        b = _tracklet(2, 110, 200, (10.8, 0.0), (20.0, 0.0))

        remap = solve_links([a, b], FPS, StitchConfig())

        assert remap[2] == remap[1]

    def test_a_chain_of_three_collapses_to_one_identity(self):
        a = _tracklet(1, 0, 100, (0.0, 0.0), (10.0, 0.0), velocity=(5.0, 0.0))
        b = _tracklet(2, 110, 200, (10.8, 0.0), (20.0, 0.0), velocity=(5.0, 0.0))
        c = _tracklet(3, 210, 300, (20.8, 0.0), (30.0, 0.0))

        remap = solve_links([a, b, c], FPS, StitchConfig())

        assert len({remap[1], remap[2], remap[3]}) == 1

    def test_one_successor_at_most_so_two_fragments_cannot_claim_the_same_continuation(self):
        # Both a1 and a2 end where b starts. Only one may take it; the other keeps its id.
        a1 = _tracklet(1, 0, 100, (0.0, 0.0), (10.0, 0.0), velocity=(5.0, 0.0))
        a2 = _tracklet(2, 0, 100, (0.0, 0.5), (10.0, 0.5), velocity=(5.0, 0.0))
        b = _tracklet(3, 110, 200, (10.8, 0.0), (20.0, 0.0))

        remap = solve_links([a1, a2, b], FPS, StitchConfig())

        assert remap[1] != remap[2]
        assert (remap[3] == remap[1]) != (remap[3] == remap[2])

    def test_nothing_is_merged_when_every_link_is_forbidden(self):
        a = _tracklet(1, 0, 100, (0.0, 0.0), (10.0, 0.0), role=TEAM_A)
        b = _tracklet(2, 110, 200, (10.1, 0.0), (20.0, 0.0), role=TEAM_B)

        remap = solve_links([a, b], FPS, StitchConfig())

        assert remap[1] != remap[2]

    def test_a_link_above_max_cost_is_refused_rather_than_taken_as_the_best_available(self):
        # The policy, in a test: prefer a new ID over a wrong merge.
        a = _tracklet(1, 0, 100, (0.0, 0.0), (10.0, 0.0), velocity=(0.0, 8.0))
        b = _tracklet(2, 110, 200, (11.4, 0.0), (20.0, 0.0))

        remap = solve_links([a, b], FPS, StitchConfig(max_link_cost=0.01))

        assert remap[1] != remap[2]

    def test_a_short_fragment_is_absorbed_across_a_small_gap(self):
        # A 2-frame fragment has no velocity of its own, but across a 0.1 s gap it does
        # not need one — the long side's geometry decides it.
        a = _tracklet(1, 0, 2, (0.0, 0.0), (1.0, 0.0), n=3)
        b = _tracklet(2, 10, 100, (1.2, 0.0), (20.0, 0.0))

        remap = solve_links([a, b], FPS, StitchConfig(min_tracklet_s=5 / FPS))

        assert remap[1] == remap[2]

    def test_a_short_fragment_is_still_refused_across_a_long_gap(self):
        a = _tracklet(1, 0, 2, (0.0, 0.0), (1.0, 0.0), n=3)
        b = _tracklet(2, 100, 200, (2.0, 0.0), (20.0, 0.0))

        remap = solve_links([a, b], FPS, StitchConfig(min_tracklet_s=5 / FPS))

        assert remap[1] != remap[2]


class TestRosterOverflow:
    def test_a_frame_with_more_than_eleven_of_one_team_is_reported(self):
        # Cardinality cannot forbid a merge — merging only ever reduces the count — so it
        # is a diagnostic for under-merging, not a constraint on the solver.
        tracklets = [_tracklet(i, 0, 100, (0.0, 0.0), (1.0, 0.0)) for i in range(13)]

        overflow = roster_overflow(tracklets, {i: i for i in range(13)}, roster_outfield=11)

        assert overflow[TEAM_A] == 2

    def test_a_legal_roster_reports_nothing(self):
        tracklets = [_tracklet(i, 0, 100, (0.0, 0.0), (1.0, 0.0)) for i in range(11)]

        assert roster_overflow(tracklets, {i: i for i in range(11)}, roster_outfield=11) == {}

    def test_merged_fragments_count_once(self):
        tracklets = [
            _tracklet(1, 0, 50, (0.0, 0.0), (1.0, 0.0)),
            _tracklet(2, 60, 100, (1.0, 0.0), (2.0, 0.0)),
        ]

        assert roster_overflow(tracklets, {1: 1, 2: 1}, roster_outfield=1) == {}


class TestApplyRemap:
    def test_one_box_per_identity_per_frame_when_a_merge_spans_a_handover_seam(self):
        # Overlap tolerance lets two tracklets sharing a few frames be joined, so both
        # can carry a box on the seam frames. Two rows with one id is invalid MOT and the
        # extra row would score as a false positive.
        tracks = _tracks({0: [(1, [10.0, 10.0, 40.0, 100.0]), (2, [14.0, 12.0, 44.0, 102.0])]})

        out = apply_remap(tracks, {1: 1, 2: 1})

        assert out[0].id.tolist() == [1]
        assert out[0].xyxy[0] == pytest.approx(
            np.array([10.0, 10.0, 40.0, 100.0], dtype=np.float32)
        )

    def test_track_ids_are_rewritten_and_frames_preserved(self):
        from football_tracker.tracking.tracker import TrackedObjects

        frames = {
            0: TrackedObjects(
                xyxy=np.array([[0.0, 0.0, 1.0, 2.0]], np.float32),
                id=np.array([5]),
                confidence=np.ones(1, np.float32),
                class_id=np.array([0]),
            )
        }

        out = apply_remap(frames, {5: 1})

        assert out[0].id.tolist() == [1]
        assert out[0].xyxy == pytest.approx(frames[0].xyxy)

    def test_an_id_with_no_remap_entry_is_left_untouched(self):
        from football_tracker.tracking.tracker import TrackedObjects

        frames = {
            0: TrackedObjects(
                xyxy=np.zeros((1, 4), np.float32),
                id=np.array([9]),
                confidence=np.ones(1, np.float32),
                class_id=np.array([0]),
            )
        }

        assert apply_remap(frames, {})[0].id.tolist() == [9]


class TestFrameRateIndependence:
    """Durations in config, frames at the point of use: a frame count would silently
    mean a different duration at another frame rate."""

    def test_the_same_duration_is_more_frames_at_a_higher_rate(self):
        from football_tracker.reid.stitch import frames_for

        assert frames_for(0.08, 60.0) == 5
        assert frames_for(0.08, 25.0) == 2

    def test_a_duration_shorter_than_one_frame_still_means_one_frame(self):
        from football_tracker.reid.stitch import frames_for

        assert frames_for(0.001, 25.0) == 1

    def test_the_gate_admits_the_same_real_time_fragment_at_either_rate(self):
        # A 0.5 s fragment: 30 frames at 60 fps, 12 at 25 fps. Both must pass a 0.08 s gate.
        cfg = StitchConfig(min_tracklet_s=0.08)
        for fps, n in ((60.0, 30), (25.0, 12)):
            a = _tracklet(1, 0, n, (0.0, 0.0), (10.0, 0.0), velocity=(5.0, 0.0), n=n)
            b = _tracklet(2, n + 10, n + 100, (10.4, 0.0), (20.0, 0.0), n=90)
            assert solve_links([a, b], fps, cfg)[2] == 1, f"failed at {fps} fps"


def _tracks(spec):
    """frame -> [(id, box)] as TrackedObjects."""
    from football_tracker.tracking.tracker import TrackedObjects

    out = {}
    for frame, rows in spec.items():
        out[frame] = TrackedObjects(
            xyxy=np.array([b for _, b in rows], dtype=np.float32).reshape(-1, 4),
            id=np.array([i for i, _ in rows], dtype=np.int64),
            confidence=np.ones(len(rows), dtype=np.float32),
            class_id=np.zeros(len(rows), dtype=np.int64),
        )
    return out


class TestSoftTeamFilter:
    def test_two_confident_disagreeing_labels_still_forbid_the_link(self):
        a = _tracklet(1, 0, 100, (0.0, 0.0), (10.0, 0.0), role=TEAM_A)
        b = _tracklet(2, 110, 200, (10.1, 0.0), (20.0, 0.0), role=TEAM_B)

        assert link_cost(a, b, FPS, StitchConfig()) is None

    def test_a_weakly_labelled_fragment_cannot_veto_a_merge_geometry_supports(self):
        # A correct merge must not be blocked by a wrong label on a short fragment.
        a = _tracklet(1, 0, 100, (0.0, 0.0), (10.0, 0.0), velocity=(5.0, 0.0), role=TEAM_A)
        b = _tracklet(2, 110, 200, (10.8, 0.0), (20.0, 0.0), role=TEAM_B)
        b = Tracklet(**{**b.__dict__, "role_confidence": 0.6})

        cost = link_cost(a, b, FPS, StitchConfig())

        assert cost is not None

    def test_linking_across_a_weak_label_is_allowed_but_not_free(self):
        a = _tracklet(1, 0, 100, (0.0, 0.0), (10.0, 0.0), velocity=(5.0, 0.0), role=TEAM_A)
        same = _tracklet(2, 110, 200, (10.8, 0.0), (20.0, 0.0), role=TEAM_A)
        other = Tracklet(**{**same.__dict__, "role": TEAM_B, "role_confidence": 0.6})

        cfg = StitchConfig()
        assert link_cost(a, other, FPS, cfg) > link_cost(a, same, FPS, cfg)

    def test_an_unlabelled_tracklet_can_still_be_stitched(self):
        a = _tracklet(1, 0, 100, (0.0, 0.0), (10.0, 0.0), velocity=(5.0, 0.0), role=Role.UNKNOWN)
        a = Tracklet(**{**a.__dict__, "role_confidence": 0.0})
        b = _tracklet(2, 110, 200, (10.8, 0.0), (20.0, 0.0), role=TEAM_A)

        assert link_cost(a, b, FPS, StitchConfig()) is not None


class TestHandoverAndShortFragments:
    def test_a_few_frames_of_overlap_is_a_handover_not_two_places_at_once(self):
        a = _tracklet(1, 0, 249, (0.0, 0.0), (10.0, 0.0), velocity=(5.0, 0.0))
        b = _tracklet(2, 245, 400, (10.1, 0.0), (20.0, 0.0))

        assert link_cost(a, b, FPS, StitchConfig(overlap_tolerance_s=0.1)) is not None

    def test_a_sustained_overlap_remains_forbidden(self):
        a = _tracklet(1, 0, 300, (0.0, 0.0), (10.0, 0.0))
        b = _tracklet(2, 100, 400, (10.1, 0.0), (20.0, 0.0))

        assert link_cost(a, b, FPS, StitchConfig(overlap_tolerance_s=0.1)) is None

    def test_a_short_fragment_is_absorbed_across_a_tiny_gap(self):
        a = _tracklet(1, 0, 100, (0.0, 0.0), (10.0, 0.0), velocity=(5.0, 0.0))
        b = _tracklet(2, 103, 105, (10.3, 0.0), (10.5, 0.0), n=3)

        assert link_cost(a, b, FPS, StitchConfig()) is not None

    def test_a_short_fragment_is_refused_across_a_long_gap(self):
        a = _tracklet(1, 0, 100, (0.0, 0.0), (10.0, 0.0), velocity=(5.0, 0.0))
        b = _tracklet(2, 200, 202, (12.0, 0.0), (12.2, 0.0), n=3)

        assert link_cost(a, b, FPS, StitchConfig(short_fragment_max_gap_s=0.25)) is None
