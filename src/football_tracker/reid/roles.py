"""Role resolution: which colour outliers are goalkeepers and which are referees.

Colour alone cannot do this: a goalkeeper's kit can sit as close to the referee's
in Lab as no threshold will separate. Pitch position can — a goalkeeper spends its life
within ~20 m of the goal line and a referee does not.

Four signals combine so no single one decides:

1. **Colour outlier status** — supplied by the caller. A track inside a team's
   rejection radius is an outfield player and never reaches this module, so a
   referee is never given a team.
2. **Goal-zone geometry** — the discriminator.
3. **Detector class evidence** — a weak prior that can lift a borderline geometric
   signal, never decide on its own.
4. **Cardinality** — at most ``max_goalkeepers``; excess candidates are demoted.

Pure: arrays and dataclasses in, dataclass out. No model, no filesystem.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from football_tracker.config import TeamIDConfig
from football_tracker.detection.interface import ObjectClass
from football_tracker.homography.pitch import PitchSpec

# Weight on the detector's class evidence relative to the geometric signal. Low by
# design: this prior nudges, it never decides.
_CLASS_EVIDENCE_WEIGHT = 0.5


class Role(StrEnum):
    """The category a track belongs to, decided once over its whole life.

    ``TEAM_A``/``TEAM_B`` are outfield players; the palette records which colour is
    which. ``UNKNOWN`` means not enough evidence — a first-class value, never a guess.
    """

    TEAM_A = "team_a"
    TEAM_B = "team_b"
    GOALKEEPER = "goalkeeper"
    REFEREE = "referee"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TrackPitchStats:
    """How much of a track's life was spent in a goal zone."""

    goal_zone_fraction: float  # share of frames within gk_zone_m of either goal line
    n_frames: int  # frames with a usable pitch position


@dataclass(frozen=True)
class RoleResolution:
    """Roles for the outlier tracks, plus the colour centroids they imply."""

    roles: dict[int, Role]
    gk_centroids: np.ndarray  # (G, 3) Lab, G in 0..max_goalkeepers
    gk_track_ids: tuple[tuple[int, ...], ...]  # member tracks per GK centroid
    referee_centroid: np.ndarray | None  # (3,) Lab, None when no referee was found


def pitch_stats(positions_m: np.ndarray, spec: PitchSpec, gk_zone_m: float) -> TrackPitchStats:
    """Summarise one track's pitch positions.

    ``positions_m`` is an ``(N, 2)`` array of foot points already projected through the
    per-frame homography. Frames without a usable calibration should simply be omitted
    by the caller rather than passed as NaN.
    """
    positions_m = np.asarray(positions_m, dtype=np.float64).reshape(-1, 2)
    if len(positions_m) == 0:
        return TrackPitchStats(goal_zone_fraction=0.0, n_frames=0)

    x = positions_m[:, 0]
    in_zone = (x < gk_zone_m) | (x > spec.length - gk_zone_m)
    return TrackPitchStats(goal_zone_fraction=float(in_zone.mean()), n_frames=len(positions_m))


def _colour_clusters(
    ids: list[int], descriptors: dict[int, np.ndarray], radius: float
) -> list[list[int]]:
    """Group outliers that share a kit, so the referees collapse into one cluster."""
    if len(ids) == 1:
        return [[ids[0]]]

    from sklearn.cluster import AgglomerativeClustering

    features = np.stack([descriptors[i] for i in ids])
    labels = AgglomerativeClustering(
        n_clusters=None, distance_threshold=radius, linkage="average"
    ).fit_predict(features)
    groups: dict[int, list[int]] = {}
    for track_id, label in zip(ids, labels, strict=True):
        groups.setdefault(int(label), []).append(track_id)
    return list(groups.values())


def _gk_score(
    members: list[int],
    stats: dict[int, TrackPitchStats],
    class_evidence: dict[int, dict[int, int]],
) -> float:
    """How strongly a colour cluster looks like a goalkeeper.

    Geometry is weighted by frame count so a 400-frame track outvotes a 5-frame one.
    A cluster with no calibrated frames at all scores 0 — it must not claim keeper on
    an absence of evidence.
    """
    weights = np.array([stats[m].n_frames for m in members], dtype=np.float64)
    if weights.sum() == 0:
        geometric = 0.0
    else:
        zone = np.array([stats[m].goal_zone_fraction for m in members])
        geometric = float(np.average(zone, weights=weights))

    gk_cls = int(ObjectClass.GOALKEEPER)
    seen = sum(sum(class_evidence.get(m, {}).values()) for m in members)
    as_gk = sum(class_evidence.get(m, {}).get(gk_cls, 0) for m in members)
    class_prior = (as_gk / seen) if seen else 0.0
    return geometric + _CLASS_EVIDENCE_WEIGHT * class_prior


def resolve_roles(
    outlier_ids: list[int],
    descriptors: dict[int, np.ndarray],
    stats: dict[int, TrackPitchStats],
    class_evidence: dict[int, dict[int, int]],
    reject_radius: float,
    cfg: TeamIDConfig,
) -> RoleResolution:
    """Split the colour outliers into goalkeepers and referees.

    ``outlier_ids`` are tracks the palette fit pushed outside every team's rejection
    radius. ``class_evidence`` maps a track to ``{ObjectClass int: frame count}`` and
    may be empty — it is a prior, never a requirement.
    """
    if not outlier_ids:
        return RoleResolution({}, np.zeros((0, 3)), (), None)

    clusters = _colour_clusters(list(outlier_ids), descriptors, reject_radius)
    scored = sorted(
        ((_gk_score(members, stats, class_evidence), members) for members in clusters),
        key=lambda pair: pair[0],
        reverse=True,
    )

    roles: dict[int, Role] = {}
    gk_centroids: list[np.ndarray] = []
    gk_track_ids: list[tuple[int, ...]] = []
    referee_members: list[int] = []

    for score, members in scored:
        qualifies = score >= cfg.gk_zone_frac and len(gk_centroids) < cfg.max_goalkeepers
        if qualifies:
            gk_centroids.append(np.mean([descriptors[m] for m in members], axis=0))
            gk_track_ids.append(tuple(members))
            for m in members:
                roles[m] = Role.GOALKEEPER
        else:
            referee_members.extend(members)
            for m in members:
                roles[m] = Role.REFEREE

    referee_centroid = (
        np.mean([descriptors[m] for m in referee_members], axis=0) if referee_members else None
    )
    return RoleResolution(
        roles=roles,
        gk_centroids=np.stack(gk_centroids) if gk_centroids else np.zeros((0, 3)),
        gk_track_ids=tuple(gk_track_ids),
        referee_centroid=referee_centroid,
    )
