"""Role resolution: goal-zone geometry, cardinality, and referee clustering.

The kit set throughout: one yellow keeper, one light-blue keeper, black referees. The
light-blue keeper sits close to the referees in colour, so the goal-zone prior has to
split them.
"""

from __future__ import annotations

import numpy as np

from football_tracker.config import TeamIDConfig
from football_tracker.detection.interface import ObjectClass
from football_tracker.homography.pitch import PitchSpec
from football_tracker.reid.roles import Role, pitch_stats, resolve_roles

SPEC = PitchSpec()
# Lab descriptors in OpenCV's 0-255 ranges.
YELLOW_GK = np.array([224.0, 115.0, 199.0])
BLUE_GK = np.array([140.0, 120.0, 95.0])  # light blue -> Lab b below neutral
BLACK_REF = np.array([90.0, 128.0, 132.0])


def _near_goal(x: float, n: int = 40) -> np.ndarray:
    return np.column_stack([np.full(n, x), np.full(n, 34.0)])


def _diagonal_run(n: int = 40) -> np.ndarray:
    """A centre referee's diagonal — spans the middle of the pitch, never a goal zone."""
    return np.column_stack([np.linspace(30.0, 75.0, n), np.linspace(12.0, 55.0, n)])


def _stats(positions: np.ndarray, cfg: TeamIDConfig):
    return pitch_stats(positions, SPEC, cfg.gk_zone_m)


# --- pitch_stats ------------------------------------------------------------------


def test_goal_zone_fraction_is_one_for_a_keeper_on_its_line():
    st = pitch_stats(_near_goal(6.0), SPEC, gk_zone_m=20.0)
    assert st.goal_zone_fraction == 1.0
    assert st.n_frames == 40


def test_goal_zone_fraction_is_one_at_the_far_goal_too():
    st = pitch_stats(_near_goal(SPEC.length - 6.0), SPEC, gk_zone_m=20.0)
    assert st.goal_zone_fraction == 1.0


def test_goal_zone_fraction_is_zero_for_a_diagonal_runner():
    assert pitch_stats(_diagonal_run(), SPEC, gk_zone_m=20.0).goal_zone_fraction == 0.0


def test_empty_track_yields_zero_frames_and_no_goal_zone_evidence():
    st = pitch_stats(np.zeros((0, 2)), SPEC, gk_zone_m=20.0)
    assert st.n_frames == 0 and st.goal_zone_fraction == 0.0


# --- resolve_roles ----------------------------------------------------------------


def test_keeper_and_referee_are_separated_by_geometry_not_colour():
    """Blue GK and black refs are close in colour, so only the goal-zone prior can split them."""
    cfg = TeamIDConfig()
    stats = {1: _stats(_near_goal(5.0), cfg), 2: _stats(_diagonal_run(), cfg)}
    desc = {1: BLUE_GK, 2: BLACK_REF}
    res = resolve_roles([1, 2], desc, stats, {}, reject_radius=30.0, cfg=cfg)
    assert res.roles[1] is Role.GOALKEEPER
    assert res.roles[2] is Role.REFEREE


def test_both_keepers_resolve_despite_different_kit_colours():
    cfg = TeamIDConfig()
    stats = {1: _stats(_near_goal(5.0), cfg), 2: _stats(_near_goal(100.0), cfg)}
    desc = {1: BLUE_GK, 2: YELLOW_GK}
    res = resolve_roles([1, 2], desc, stats, {}, reject_radius=30.0, cfg=cfg)
    assert res.roles[1] is Role.GOALKEEPER and res.roles[2] is Role.GOALKEEPER
    assert res.gk_centroids.shape == (2, 3)


def test_detector_class_evidence_breaks_a_weak_geometric_signal():
    cfg = TeamIDConfig()
    # Half its life inside the zone: below gk_zone_frac on geometry alone.
    pos = np.column_stack([np.linspace(15.0, 26.0, 40), np.full(40, 34.0)])
    stats = {1: _stats(pos, cfg)}
    evidence = {1: {int(ObjectClass.GOALKEEPER): 30, int(ObjectClass.PLAYER): 5}}
    res = resolve_roles([1], {1: YELLOW_GK}, stats, evidence, reject_radius=30.0, cfg=cfg)
    assert res.roles[1] is Role.GOALKEEPER


def test_third_keeper_candidate_is_demoted_by_the_cardinality_prior():
    cfg = TeamIDConfig()
    stats, desc = {}, {}
    for i, x in enumerate((5.0, 100.0, 8.0)):
        stats[i] = _stats(_near_goal(x), cfg)
        desc[i] = np.array([224.0 - 40 * i, 115.0 + 5 * i, 199.0 - 50 * i])
    res = resolve_roles([0, 1, 2], desc, stats, {}, reject_radius=20.0, cfg=cfg)
    assert sum(r is Role.GOALKEEPER for r in res.roles.values()) == cfg.max_goalkeepers


def test_referees_sharing_the_black_kit_collapse_to_one_centroid():
    cfg = TeamIDConfig()
    stats, desc = {}, {}
    for i in range(3):
        stats[i] = _stats(_diagonal_run(), cfg)
        desc[i] = BLACK_REF + np.array([2.0 * i, 0.0, 1.0 * i])
    res = resolve_roles([0, 1, 2], desc, stats, {}, reject_radius=30.0, cfg=cfg)
    assert all(r is Role.REFEREE for r in res.roles.values())
    assert res.referee_centroid is not None and res.referee_centroid.shape == (3,)
    assert res.gk_centroids.shape == (0, 3)


def test_full_scene_two_keepers_and_three_referees():
    cfg = TeamIDConfig()
    stats, desc = {}, {}
    stats[10], desc[10] = _stats(_near_goal(5.0), cfg), BLUE_GK
    stats[11], desc[11] = _stats(_near_goal(100.0), cfg), YELLOW_GK
    for i in (20, 21, 22):
        stats[i], desc[i] = _stats(_diagonal_run(), cfg), BLACK_REF + np.array([i - 20.0, 0, 0])
    res = resolve_roles(list(desc), desc, stats, {}, reject_radius=30.0, cfg=cfg)
    assert res.roles[10] is Role.GOALKEEPER and res.roles[11] is Role.GOALKEEPER
    assert all(res.roles[i] is Role.REFEREE for i in (20, 21, 22))


def test_no_outliers_yields_an_empty_resolution():
    res = resolve_roles([], {}, {}, {}, reject_radius=30.0, cfg=TeamIDConfig())
    assert res.roles == {}
    assert res.referee_centroid is None
    assert res.gk_centroids.shape == (0, 3)


def test_a_track_with_no_pitch_evidence_falls_back_to_referee_not_keeper():
    """No calibration for its frames => no goal-zone evidence => must not claim GK."""
    cfg = TeamIDConfig()
    stats = {1: pitch_stats(np.zeros((0, 2)), SPEC, cfg.gk_zone_m)}
    res = resolve_roles([1], {1: BLUE_GK}, stats, {}, reject_radius=30.0, cfg=cfg)
    assert res.roles[1] is Role.REFEREE


def test_every_outlier_receives_exactly_one_role():
    cfg = TeamIDConfig()
    stats, desc = {}, {}
    for i in range(6):
        stats[i] = _stats(_diagonal_run() if i % 2 else _near_goal(5.0), cfg)
        desc[i] = BLACK_REF + np.array([3.0 * i, 0.0, 0.0])
    res = resolve_roles(list(desc), desc, stats, {}, reject_radius=10.0, cfg=cfg)
    assert set(res.roles) == set(desc)
    assert all(r in (Role.GOALKEEPER, Role.REFEREE) for r in res.roles.values())
