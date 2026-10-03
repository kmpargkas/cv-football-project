"""Palette fitting and team assignment.

The kit set throughout: red and white outfield, a light-blue and a yellow keeper, black
referees.
"""

from __future__ import annotations

import numpy as np
import pytest

from football_tracker.config import TeamIDConfig
from football_tracker.homography.pitch import PitchSpec
from football_tracker.reid.roles import Role, pitch_stats
from football_tracker.reid.teamid import (
    NEUTRAL_BGR,
    KitColours,
    TeamAssigner,
    TeamPalette,
    TrackLabel,
    TrackObservation,
    canonicalise_palette,
    describe_team_order,
    fit_palette,
    lab_to_bgr,
)

SPEC = PitchSpec()
RED = np.array([140.0, 165.0, 149.0])
WHITE = np.array([243.0, 123.0, 134.0])
YELLOW_GK = np.array([224.0, 115.0, 199.0])
BLUE_GK = np.array([140.0, 120.0, 95.0])
BLACK_REF = np.array([90.0, 128.0, 132.0])


def _outfield_positions(n=60):
    return np.column_stack([np.linspace(30.0, 75.0, n), np.linspace(20.0, 48.0, n)])


def _goal_positions(x, n=60):
    return np.column_stack([np.full(n, x), np.full(n, 34.0)])


def _obs(descriptor, positions, cfg, n_samples=12, n_frames=200, class_counts=None):
    return TrackObservation(
        descriptor=np.asarray(descriptor, dtype=np.float64),
        n_samples=n_samples,
        n_frames=n_frames,
        stats=pitch_stats(positions, SPEC, cfg.gk_zone_m),
        class_counts=class_counts or {},
    )


def _scene(cfg, seed=0, n_per_team=10):
    """Twenty outfield players, two keepers, three referees."""
    rng = np.random.default_rng(seed)
    obs = {}
    for i in range(n_per_team):
        obs[i] = _obs(RED + rng.normal(0, 6, 3), _outfield_positions(), cfg)
    for i in range(n_per_team, 2 * n_per_team):
        obs[i] = _obs(WHITE + rng.normal(0, 6, 3), _outfield_positions(), cfg)
    obs[100] = _obs(BLUE_GK, _goal_positions(5.0), cfg)
    obs[101] = _obs(YELLOW_GK, _goal_positions(100.0), cfg)
    for j, rid in enumerate((200, 201, 202)):
        obs[rid] = _obs(BLACK_REF + np.array([2.0 * j, 0, 0]), _outfield_positions(), cfg)
    return obs


def _vote(assigner, track_id, descriptor, times=6):
    for _ in range(times):
        assigner.observe(track_id, np.asarray(descriptor, np.float32))


# --- fit_palette ------------------------------------------------------------------


def test_reject_radius_lands_between_in_team_scatter_and_the_anomalies():
    """In-team scatter is 5-20 Lab units; anomalies sit at 47-68."""
    cfg = TeamIDConfig()
    palette = fit_palette(_scene(cfg), cfg)
    assert 20.0 <= palette.reject_radius < 47.0


def test_refit_recovers_the_true_kit_means():
    cfg = TeamIDConfig()
    palette = fit_palette(_scene(cfg), cfg)
    for truth in (RED, WHITE):
        assert np.min(np.linalg.norm(palette.team_centroids - truth, axis=1)) < 8.0


def test_keepers_and_referees_never_become_team_centroids():
    cfg = TeamIDConfig()
    palette = fit_palette(_scene(cfg), cfg)
    for anomaly in (BLUE_GK, YELLOW_GK, BLACK_REF):
        assert np.min(np.linalg.norm(palette.team_centroids - anomaly, axis=1)) > 30.0


def test_both_keepers_and_the_referee_kit_are_recovered():
    cfg = TeamIDConfig()
    palette = fit_palette(_scene(cfg), cfg)
    assert palette.gk_centroids.shape == (2, 3)
    assert palette.referee_centroid is not None


def test_short_lived_tracks_are_excluded_from_the_fit():
    """A high-vis steward leaking past the pitch mask would pollute a yellow keeper's
    centroid silently, so brief tracks cannot inform the fit."""
    cfg = TeamIDConfig()
    obs = _scene(cfg)
    steward = _obs(YELLOW_GK, _outfield_positions(), cfg, n_samples=2, n_frames=4)
    obs[999] = steward
    palette = fit_palette(obs, cfg)
    assert 999 not in palette.provenance["fitted_track_ids"]


def test_provenance_records_what_the_fit_saw():
    cfg = TeamIDConfig()
    palette = fit_palette(_scene(cfg), cfg)
    assert palette.provenance["n_tracks"] == 25
    assert palette.provenance["silhouette"] > 0.5
    assert len(palette.provenance["fitted_track_ids"]) == 25


def test_fit_needs_at_least_two_eligible_tracks():
    cfg = TeamIDConfig()
    with pytest.raises(ValueError, match="eligible"):
        fit_palette({0: _obs(RED, _outfield_positions(), cfg)}, cfg)


# --- TeamAssigner -----------------------------------------------------------------


def test_referees_are_never_given_a_team():
    """The hard requirement: a referee in a team colour is a scoring error."""
    cfg = TeamIDConfig()
    obs = _scene(cfg)
    assigner = TeamAssigner(fit_palette(obs, cfg), cfg)
    for tid, o in obs.items():
        _vote(assigner, tid, o.descriptor)
    labels = assigner.consolidate()
    for rid in (200, 201, 202):
        assert labels[rid].role is Role.REFEREE
        assert labels[rid].team == -1


def test_outfield_players_split_into_exactly_two_teams():
    cfg = TeamIDConfig()
    obs = _scene(cfg)
    assigner = TeamAssigner(fit_palette(obs, cfg), cfg)
    for tid, o in obs.items():
        _vote(assigner, tid, o.descriptor)
    labels = assigner.consolidate()
    reds = {labels[i].team for i in range(10)}
    whites = {labels[i].team for i in range(10, 20)}
    assert len(reds) == 1 and len(whites) == 1 and reds != whites
    assert all(labels[i].role in (Role.TEAM_A, Role.TEAM_B) for i in range(20))


def test_keepers_are_labelled_goalkeeper_not_a_team():
    cfg = TeamIDConfig()
    obs = _scene(cfg)
    assigner = TeamAssigner(fit_palette(obs, cfg), cfg)
    for tid, o in obs.items():
        _vote(assigner, tid, o.descriptor)
    labels = assigner.consolidate()
    assert labels[100].role is Role.GOALKEEPER
    assert labels[101].role is Role.GOALKEEPER


def test_a_track_with_too_few_samples_is_unknown_not_guessed():
    cfg = TeamIDConfig()
    assigner = TeamAssigner(fit_palette(_scene(cfg), cfg), cfg)
    _vote(assigner, 777, RED, times=cfg.min_samples_assign - 1)
    assert assigner.label(777).role is Role.UNKNOWN


def test_an_ambiguous_sample_between_two_centroids_casts_no_vote():
    cfg = TeamIDConfig()
    palette = fit_palette(_scene(cfg), cfg)
    assigner = TeamAssigner(palette, cfg)
    _vote(assigner, 888, palette.team_centroids.mean(axis=0), times=10)
    label = assigner.label(888)
    assert label.role is Role.UNKNOWN
    assert label.n_samples == 0  # every vote was refused by the margin guard


def test_rejected_samples_are_counted_not_silently_dropped():
    cfg = TeamIDConfig()
    assigner = TeamAssigner(fit_palette(_scene(cfg), cfg), cfg)
    for _ in range(4):
        assigner.observe(555, None)
    label = assigner.label(555)
    assert label.n_rejected == 4
    assert label.role is Role.UNKNOWN


def test_an_unseen_track_is_unknown():
    cfg = TeamIDConfig()
    assigner = TeamAssigner(fit_palette(_scene(cfg), cfg), cfg)
    assert assigner.label(31337).role is Role.UNKNOWN


def test_every_track_gets_exactly_one_role_for_its_whole_life():
    """The structural invariant: a track can never flip role mid-life."""
    cfg = TeamIDConfig()
    obs = _scene(cfg)
    assigner = TeamAssigner(fit_palette(obs, cfg), cfg)
    for tid, o in obs.items():
        _vote(assigner, tid, o.descriptor)
    labels = assigner.consolidate()
    assert set(labels) == set(obs)
    assert all(isinstance(v.role, Role) for v in labels.values())


def test_a_majority_of_red_votes_survives_a_few_contaminated_samples():
    cfg = TeamIDConfig()
    palette = fit_palette(_scene(cfg), cfg)
    assigner = TeamAssigner(palette, cfg)
    _vote(assigner, 42, RED, times=8)
    _vote(assigner, 42, WHITE, times=2)  # occlusion contamination
    label = assigner.label(42)
    assert label.role in (Role.TEAM_A, Role.TEAM_B)
    assert label.confidence == pytest.approx(0.8, abs=0.01)


def test_confidence_is_the_winning_vote_share():
    cfg = TeamIDConfig()
    assigner = TeamAssigner(fit_palette(_scene(cfg), cfg), cfg)
    _vote(assigner, 7, RED, times=10)
    assert assigner.label(7).confidence == pytest.approx(1.0)


def test_a_brief_track_cannot_claim_a_rare_role():
    """At most two keepers and a handful of referees against twenty outfield players:
    claiming a rare role needs the same evidence that fit participation needs."""
    cfg = TeamIDConfig()
    obs = _scene(cfg)
    assigner = TeamAssigner(fit_palette(obs, cfg), cfg)
    for tid, o in obs.items():
        _vote(assigner, tid, o.descriptor)
    _vote(assigner, 41, YELLOW_GK, times=5)  # brief, but enough to clear min_samples_assign

    labels = assigner.consolidate(track_frames={**{t: 200 for t in obs}, 41: 5})
    assert labels[41].role is Role.UNKNOWN
    assert labels[100].role is Role.GOALKEEPER  # the genuine keepers are untouched
    assert labels[101].role is Role.GOALKEEPER


def test_a_brief_track_may_still_be_assigned_a_team():
    """The rare-role gate must not swallow ordinary outfield players."""
    cfg = TeamIDConfig()
    obs = _scene(cfg)
    assigner = TeamAssigner(fit_palette(obs, cfg), cfg)
    for tid, o in obs.items():
        _vote(assigner, tid, o.descriptor)
    _vote(assigner, 42, RED, times=5)

    labels = assigner.consolidate(track_frames={**{t: 200 for t in obs}, 42: 5})
    assert labels[42].role in (Role.TEAM_A, Role.TEAM_B)


def test_consolidate_without_lifetimes_keeps_the_online_behaviour():
    """`label()` is the online best-effort view; whole-clip rules need whole-clip data."""
    cfg = TeamIDConfig()
    obs = _scene(cfg)
    assigner = TeamAssigner(fit_palette(obs, cfg), cfg)
    _vote(assigner, 41, YELLOW_GK, times=5)
    assert assigner.consolidate()[41].role is Role.GOALKEEPER


def test_palette_round_trips_through_its_own_dict_form():
    cfg = TeamIDConfig()
    palette = fit_palette(_scene(cfg), cfg)
    restored = TeamPalette.from_dict(palette.to_dict())
    assert np.allclose(restored.team_centroids, palette.team_centroids)
    assert np.allclose(restored.gk_centroids, palette.gk_centroids)
    assert np.allclose(restored.referee_centroid, palette.referee_centroid)
    assert restored.reject_radius == palette.reject_radius


def test_palette_round_trips_when_there_is_no_referee():
    cfg = TeamIDConfig()
    obs = {k: v for k, v in _scene(cfg).items() if k < 200}
    palette = fit_palette(obs, cfg)
    restored = TeamPalette.from_dict(palette.to_dict())
    assert restored.referee_centroid is None


# --- build_observations -----------------------------------------------------------


def test_build_observations_uses_a_median_not_a_mean():
    """A few contaminated frames must not move the per-track descriptor."""
    from football_tracker.reid.teamid import build_observations

    cfg = TeamIDConfig()
    labs = [RED] * 8 + [WHITE] * 2
    obs = build_observations({1: labs}, {}, {}, {1: 200}, SPEC, cfg)
    assert np.linalg.norm(obs[1].descriptor - RED) < 5.0
    assert obs[1].n_samples == 10
    assert obs[1].n_frames == 200


def test_build_observations_skips_tracks_with_no_accepted_sample():
    from football_tracker.reid.teamid import build_observations

    assert build_observations({1: []}, {}, {}, {1: 200}, SPEC, TeamIDConfig()) == {}


def test_build_observations_carries_goal_zone_evidence():
    from football_tracker.reid.teamid import build_observations

    cfg = TeamIDConfig()
    obs = build_observations(
        {1: [YELLOW_GK]}, {1: list(_goal_positions(5.0))}, {}, {1: 200}, SPEC, cfg
    )
    assert obs[1].stats.goal_zone_fraction == 1.0


def test_build_observations_without_positions_yields_no_goal_zone_evidence():
    """No usable homography for a track must not let it claim goalkeeper."""
    from football_tracker.reid.teamid import build_observations

    obs = build_observations({1: [YELLOW_GK]}, {}, {}, {1: 200}, SPEC, TeamIDConfig())
    assert obs[1].stats.n_frames == 0
    assert obs[1].stats.goal_zone_fraction == 0.0


def test_build_observations_falls_back_to_sample_count_for_lifetime():
    from football_tracker.reid.teamid import build_observations

    obs = build_observations({1: [RED, RED, RED]}, {}, {}, {}, SPEC, TeamIDConfig())
    assert obs[1].n_frames == 3


def test_a_noisy_team_member_does_not_pollute_the_referee_centroid():
    """An outlier becomes a referee or keeper only if its colour is closer to that role
    than to any team; otherwise it is a noisy team member and stays out of the role
    centroids."""
    cfg = TeamIDConfig()
    obs = _scene(cfg)
    # Beyond the rejection radius from both kits (~50 Lab units from white), but still
    # far nearer white than the black referee kit (~155 away).
    obs[26] = _obs(WHITE + np.array([0.0, 0.0, 50.0]), _outfield_positions(), cfg)
    palette = fit_palette(obs, cfg)

    assert 26 in palette.provenance["outlier_track_ids"]
    assert np.linalg.norm(palette.referee_centroid - BLACK_REF) < 5.0
    assert 26 not in palette.provenance["role_track_ids"]


def test_a_genuine_referee_still_reaches_the_referee_centroid():
    """The consistency pass must not swallow real referees."""
    cfg = TeamIDConfig()
    palette = fit_palette(_scene(cfg), cfg)
    assert np.linalg.norm(palette.referee_centroid - BLACK_REF) < 5.0
    assert set(palette.provenance["role_track_ids"]) == {100, 101, 200, 201, 202}


class TestSilhouetteGuard:
    """A clip whose two kits are not separable must fail loudly, not confidently."""

    def _observations(self, centres, cfg, n=8, spread=1.0):
        from football_tracker.reid.roles import TrackPitchStats

        rng = np.random.default_rng(0)
        obs = {}
        for i, centre in enumerate(centres):
            for k in range(n):
                point = np.asarray(centre, float) + rng.normal(0, spread, 3)
                obs[i * n + k] = TrackObservation(
                    descriptor=point,
                    n_samples=cfg.min_samples_fit,
                    n_frames=cfg.min_track_frames_fit,
                    stats=TrackPitchStats(goal_zone_fraction=0.0, n_frames=100),
                    class_counts={},
                )
        return obs

    def test_two_well_separated_kits_fit_normally(self):
        from football_tracker.config import TeamIDConfig
        from football_tracker.reid.teamid import fit_palette

        cfg = TeamIDConfig()
        palette = fit_palette(self._observations([(140, 166, 150), (233, 123, 135)], cfg), cfg)

        assert palette.provenance["silhouette"] >= cfg.min_silhouette

    def test_two_similar_kits_raise_instead_of_returning_a_confident_wrong_palette(self):
        import pytest

        from football_tracker.config import TeamIDConfig
        from football_tracker.reid.teamid import fit_palette

        cfg = TeamIDConfig()
        # One blob, not two teams: any k=2 split of it is arbitrary.
        obs = self._observations([(150, 140, 140), (151, 141, 141)], cfg, spread=6.0)

        with pytest.raises(ValueError, match="not separable"):
            fit_palette(obs, cfg)

    def test_the_floor_is_configurable_for_a_clip_that_justifies_it(self):
        from football_tracker.config import TeamIDConfig
        from football_tracker.reid.teamid import fit_palette

        cfg = TeamIDConfig(min_silhouette=0.0)
        obs = self._observations([(150, 140, 140), (151, 141, 141)], cfg, spread=6.0)

        assert fit_palette(obs, cfg) is not None


# --- build_targets ------------------------------------------------------------------


def test_build_targets_gives_each_keeper_its_kit_index():
    """A keeper's `team` is its row in gk_centroids, so renderers can draw its own kit."""
    palette = TeamPalette(
        team_centroids=np.stack([RED, WHITE]),
        gk_centroids=np.stack([BLUE_GK, YELLOW_GK]),
        referee_centroid=BLACK_REF,
        reject_radius=40.0,
        provenance={},
    )
    _, labels = TeamAssigner.build_targets(palette)
    assert labels == [
        (Role.TEAM_A, 0),
        (Role.TEAM_B, 1),
        (Role.GOALKEEPER, 0),
        (Role.GOALKEEPER, 1),
        (Role.REFEREE, -1),
    ]


# --- canonicalise_palette ---------------------------------------------------------
#
# KMeans cluster order is arbitrary, and everything downstream reads index 0 as
# `team_a`, so an unlucky fit would swap both teams with no error raised. The order
# is fixed by the footage instead: index 0 is the lower kit on the Lab channel where
# the two kits differ most.

# Red and blue at matched lightness: L is a tie, b is the widest channel.
BLUE = np.array([140.0, 130.0, 80.0])


def _palette(first, second):
    return TeamPalette(
        team_centroids=np.stack([first, second]),
        gk_centroids=np.stack([BLUE_GK, YELLOW_GK]),
        referee_centroid=BLACK_REF,
        reject_radius=40.0,
        provenance={},
    )


def test_canonical_order_is_independent_of_the_order_the_clusters_came_out_in():
    forward = canonicalise_palette(_palette(RED, WHITE))
    reversed_ = canonicalise_palette(_palette(WHITE, RED))

    assert np.allclose(forward.team_centroids, reversed_.team_centroids)
    assert np.allclose(forward.team_centroids[0], RED), "the darker kit is team_a"


def test_canonical_order_sorts_on_the_widest_channel():
    """Red vs blue at matched lightness: a tie on L must not decide it."""
    palette = canonicalise_palette(_palette(RED, BLUE))

    assert np.allclose(palette.team_centroids[0], BLUE), "the bluer kit is team_a"
    assert palette.provenance["canonical_order"]["channel"] == "b"


def test_canonical_order_is_idempotent():
    once = canonicalise_palette(_palette(WHITE, RED))
    twice = canonicalise_palette(once)

    assert np.allclose(once.team_centroids, twice.team_centroids)
    assert once.provenance == twice.provenance


def test_canonical_order_records_channel_and_margin():
    palette = canonicalise_palette(_palette(WHITE, RED))

    record = palette.provenance["canonical_order"]
    assert record["channel"] == "L"
    assert record["margin"] == pytest.approx(WHITE[0] - RED[0])


def test_canonical_order_keeps_the_other_centroids_intact():
    palette = canonicalise_palette(_palette(WHITE, RED))

    assert np.allclose(palette.gk_centroids, np.stack([BLUE_GK, YELLOW_GK]))
    assert np.allclose(palette.referee_centroid, BLACK_REF)
    assert palette.reject_radius == 40.0


def test_fit_palette_returns_a_canonical_palette():
    cfg = TeamIDConfig()
    palette = fit_palette(_scene(cfg), cfg)

    assert "canonical_order" in palette.provenance
    assert palette.team_centroids[0][0] < palette.team_centroids[1][0], "darker kit first"


def test_describe_team_order_names_the_kit_in_words():
    assert describe_team_order(canonicalise_palette(_palette(WHITE, RED))).startswith(
        "team_a is the darker kit (L 140 vs 243"
    )
    assert describe_team_order(canonicalise_palette(_palette(RED, BLUE))).startswith(
        "team_a is the bluer kit (b 80 vs 149"
    )


# --- lab_to_bgr ---------------------------------------------------------------------


@pytest.mark.parametrize("bgr", [(40, 40, 214), (232, 232, 232), (137, 120, 121), (0, 0, 0)])
def test_lab_to_bgr_inverts_cv2_lab_within_rounding(bgr: tuple[int, int, int]):
    import cv2

    lab = cv2.cvtColor(np.array([[bgr]], dtype=np.uint8), cv2.COLOR_BGR2LAB)[0, 0]
    out = lab_to_bgr(lab.astype(np.float64))
    assert max(abs(x - y) for x, y in zip(out, bgr, strict=True)) <= 2


def test_lab_to_bgr_returns_plain_ints_for_cv2():
    out = lab_to_bgr(np.array([140.0, 165.0, 149.0]))
    assert all(type(c) is int for c in out) and all(0 <= c <= 255 for c in out)


# --- kit colours for rendering -----------------------------------------------------


def _kit_palette(with_referee: bool = True) -> TeamPalette:
    return TeamPalette(
        team_centroids=np.array([[234.0, 123.0, 135.0], [141.0, 167.0, 150.0]]),
        gk_centroids=np.array([[226.0, 115.0, 200.0]]),
        referee_centroid=np.array([85.0, 126.0, 137.0]) if with_referee else None,
        reject_radius=20.0,
        provenance={},
    )


def _kit_label(role: Role, team: int = -1) -> TrackLabel:
    return TrackLabel(role=role, team=team, confidence=1.0, n_samples=10, n_rejected=0)


def test_kit_colours_map_each_role_to_its_own_centroid():
    palette = _kit_palette()
    colours = KitColours.from_palette(palette)
    assert colours.for_label(_kit_label(Role.TEAM_A, 0)) == lab_to_bgr(palette.team_centroids[0])
    assert colours.for_label(_kit_label(Role.TEAM_B, 1)) == lab_to_bgr(palette.team_centroids[1])
    assert colours.for_label(_kit_label(Role.GOALKEEPER, 0)) == lab_to_bgr(palette.gk_centroids[0])
    assert colours.for_label(_kit_label(Role.REFEREE)) == lab_to_bgr(palette.referee_centroid)


def test_kit_colours_fall_back_to_neutral_where_the_palette_has_no_centroid():
    colours = KitColours.from_palette(_kit_palette(with_referee=False))
    assert colours.for_label(_kit_label(Role.REFEREE)) == NEUTRAL_BGR
    assert colours.for_label(_kit_label(Role.GOALKEEPER, -1)) == NEUTRAL_BGR  # index unset
    assert colours.for_label(_kit_label(Role.GOALKEEPER, 5)) == NEUTRAL_BGR  # out of range
    assert colours.for_label(_kit_label(Role.UNKNOWN)) == NEUTRAL_BGR
    assert colours.for_label(None) == NEUTRAL_BGR


def test_kit_colours_never_give_a_referee_a_team_colour():
    colours = KitColours.from_palette(_kit_palette())
    assert colours.for_label(_kit_label(Role.REFEREE)) not in colours.team
