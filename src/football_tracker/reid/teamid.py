"""Team palette fitting and per-track team/role assignment.

The fit is **outlier-first**, and that ordering is the whole design. Torso Lab medians
separate two kits at silhouette 0.789 with k=2, but goalkeepers and referees are
absorbed into the teams at k=2 *and* k=3, only emerging at k=4.

The rejection radius is fitted from the data (median + k*MAD of within-cluster
distances), never hardcoded, so it survives a different clip's lighting.

Assignment against a frozen palette is a nearest-centroid lookup with two guards: a
**margin guard** (a sample between two centroids casts no vote) and an **evidence
guard** (a track with too few votes is ``unknown``, never a guess).

Fitting once and freezing, rather than clustering per frame, keeps assignment
deterministic and order-independent.

Pure: arrays and dataclasses in, dataclasses out. No model, no filesystem, no video.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from football_tracker.config import TeamIDConfig
from football_tracker.homography.pitch import PitchSpec
from football_tracker.reid.roles import (
    Role,
    RoleResolution,
    TrackPitchStats,
    pitch_stats,
    resolve_roles,
)

_KMEANS_SEED = 0


def lab_to_bgr(lab: np.ndarray) -> tuple[int, int, int]:
    """A palette centroid (OpenCV 0-255 CIELAB) -> cv2's (B, G, R)."""
    import cv2

    pixel = np.asarray(lab, dtype=np.float64).reshape(1, 1, 3).round().clip(0, 255).astype(np.uint8)
    b, g, r = cv2.cvtColor(pixel, cv2.COLOR_LAB2BGR)[0, 0]
    return int(b), int(g), int(r)


_LAB_CHANNELS = ("L", "a", "b")
# Either end of each Lab channel, in words: (lower, higher).
_CHANNEL_WORDS = (("darker", "lighter"), ("greener", "redder"), ("bluer", "yellower"))


def _widest_channel(team_centroids: np.ndarray) -> int:
    return int(np.argmax(np.abs(team_centroids[1] - team_centroids[0])))


def canonicalise_palette(palette: TeamPalette) -> TeamPalette:
    """Order the two kits so index 0 is the lower one on the Lab channel where they differ most.

    KMeans returns its clusters in arbitrary order and everything downstream reads
    index 0 as ``team_a``. Sorting on the widest channel makes that order a property
    of the footage alone, and the one least likely to flip on a refit.
    """
    channel = _widest_channel(palette.team_centroids)
    order = np.argsort(palette.team_centroids[:, channel], kind="stable")
    return TeamPalette(
        team_centroids=palette.team_centroids[order],
        gk_centroids=palette.gk_centroids,
        referee_centroid=palette.referee_centroid,
        reject_radius=palette.reject_radius,
        provenance={
            **palette.provenance,
            "canonical_order": {
                "channel": _LAB_CHANNELS[channel],
                "margin": float(abs(np.diff(palette.team_centroids[:, channel])[0])),
            },
        },
    )


def describe_team_order(palette: TeamPalette) -> str:
    """Which kit is ``team_a``, in words, for the console and the possession names."""
    channel = _widest_channel(palette.team_centroids)
    a, b = palette.team_centroids[:, channel]
    word = _CHANNEL_WORDS[channel][0 if a <= b else 1]
    return (
        f"team_a is the {word} kit ({_LAB_CHANNELS[channel]} {a:.0f} vs {b:.0f}, "
        f"the widest Lab channel)"
    )


@dataclass(frozen=True)
class TrackObservation:
    """Everything the palette fit knows about one track."""

    descriptor: np.ndarray  # (3,) aggregated Lab median over accepted samples
    n_samples: int  # accepted torso samples
    n_frames: int  # track lifetime in frames
    stats: TrackPitchStats  # goal-zone occupancy
    class_counts: dict[int, int] = field(default_factory=dict)  # ObjectClass -> frames


@dataclass(frozen=True)
class TrackLabel:
    """The final team/role decision for one track, over its whole life."""

    role: Role
    team: int  # outfield: 0 or 1; goalkeeper: its row in gk_centroids; -1 otherwise
    confidence: float  # winning vote share
    n_samples: int  # votes that counted
    n_rejected: int  # samples refused by a quality or margin gate


@dataclass(frozen=True)
class TeamPalette:
    """Frozen colour model for one clip: where each category sits in Lab space."""

    team_centroids: np.ndarray  # (2, 3)
    gk_centroids: np.ndarray  # (G, 3), G in 0..max_goalkeepers
    referee_centroid: np.ndarray | None  # (3,)
    reject_radius: float
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form (see ``tracking/artifacts.py`` for the file itself)."""
        return {
            "team_centroids": self.team_centroids.tolist(),
            "gk_centroids": self.gk_centroids.tolist(),
            "referee_centroid": (
                None if self.referee_centroid is None else self.referee_centroid.tolist()
            ),
            "reject_radius": float(self.reject_radius),
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TeamPalette:
        referee = data.get("referee_centroid")
        return cls(
            team_centroids=np.asarray(data["team_centroids"], dtype=np.float64).reshape(-1, 3),
            gk_centroids=np.asarray(data["gk_centroids"], dtype=np.float64).reshape(-1, 3),
            referee_centroid=None if referee is None else np.asarray(referee, dtype=np.float64),
            reject_radius=float(data["reject_radius"]),
            provenance=data.get("provenance", {}),
        )


BGR = tuple[int, int, int]
NEUTRAL_BGR: BGR = (128, 128, 128)


@dataclass(frozen=True)
class KitColours:
    """BGR per role from a clip's palette; neutral grey wherever it has no centroid."""

    team: tuple[BGR, BGR]
    goalkeepers: tuple[BGR, ...]
    referee: BGR

    @classmethod
    def from_palette(cls, palette: TeamPalette) -> KitColours:
        return cls(
            team=(lab_to_bgr(palette.team_centroids[0]), lab_to_bgr(palette.team_centroids[1])),
            goalkeepers=tuple(lab_to_bgr(c) for c in palette.gk_centroids),
            referee=(
                NEUTRAL_BGR
                if palette.referee_centroid is None
                else lab_to_bgr(palette.referee_centroid)
            ),
        )

    def for_label(self, label: TrackLabel | None) -> BGR:
        if label is None:
            return NEUTRAL_BGR
        if label.role is Role.TEAM_A:
            return self.team[0]
        if label.role is Role.TEAM_B:
            return self.team[1]
        if label.role is Role.GOALKEEPER and 0 <= label.team < len(self.goalkeepers):
            return self.goalkeepers[label.team]
        if label.role is Role.REFEREE:
            return self.referee
        return NEUTRAL_BGR


def build_observations(
    lab_samples: Mapping[int, Sequence[np.ndarray]],
    pitch_positions: Mapping[int, Sequence[np.ndarray]],
    class_counts: Mapping[int, Mapping[int, int]],
    track_frames: Mapping[int, int],
    spec: PitchSpec,
    cfg: TeamIDConfig,
) -> dict[int, TrackObservation]:
    """Fold per-frame evidence into one :class:`TrackObservation` per track.

    Tracks with no accepted sample are omitted. ``pitch_positions`` should already
    exclude frames with no usable homography.
    """
    observations: dict[int, TrackObservation] = {}
    for track_id, labs in lab_samples.items():
        if not len(labs):
            continue
        positions = pitch_positions.get(track_id, ())
        stacked = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
        observations[track_id] = TrackObservation(
            descriptor=np.median(np.stack(labs), axis=0).astype(np.float64),
            n_samples=len(labs),
            n_frames=int(track_frames.get(track_id, len(labs))),
            stats=pitch_stats(stacked, spec, cfg.gk_zone_m),
            class_counts=dict(class_counts.get(track_id, {})),
        )
    return observations


def _eligible(observations: dict[int, TrackObservation], cfg: TeamIDConfig) -> list[int]:
    """Tracks with enough samples and lifetime to inform the fit."""
    return [
        tid
        for tid, obs in observations.items()
        if obs.n_samples >= cfg.min_samples_fit and obs.n_frames >= cfg.min_track_frames_fit
    ]


def _fit_radius(features: np.ndarray, centroids: np.ndarray, cfg: TeamIDConfig) -> float:
    """Rejection radius from pooled within-cluster distances: median + k * MAD."""
    distances = np.min(np.linalg.norm(features[:, None, :] - centroids[None, :, :], axis=2), axis=1)
    median = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median)))
    return max(cfg.min_reject_radius, median + cfg.reject_k_mad * mad)


def _resolve(
    outlier_ids: list[int],
    observations: dict[int, TrackObservation],
    radius: float,
    cfg: TeamIDConfig,
) -> RoleResolution:
    """Run role resolution over a subset of the outliers."""
    return resolve_roles(
        outlier_ids=outlier_ids,
        descriptors={t: observations[t].descriptor for t in outlier_ids},
        stats={t: observations[t].stats for t in outlier_ids},
        class_evidence={t: observations[t].class_counts for t in outlier_ids},
        reject_radius=radius,
        cfg=cfg,
    )


def _role_distance(descriptor: np.ndarray, track_id: int, resolution: RoleResolution) -> float:
    """Distance from a track to the centroid of the role it was assigned."""
    if resolution.roles.get(track_id) is Role.GOALKEEPER:
        for members, centroid in zip(resolution.gk_track_ids, resolution.gk_centroids, strict=True):
            if track_id in members:
                return float(np.linalg.norm(centroid - descriptor))
    if resolution.referee_centroid is not None:
        return float(np.linalg.norm(resolution.referee_centroid - descriptor))
    return float("inf")


def fit_palette(observations: dict[int, TrackObservation], cfg: TeamIDConfig) -> TeamPalette:
    """Fit the frozen colour model for one clip.

    Raises ``ValueError`` when fewer than two tracks clear the eligibility gates or
    when the two kits do not separate (silhouette below ``min_silhouette``).
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    ids = _eligible(observations, cfg)
    if len(ids) < 2:
        raise ValueError(
            f"need at least 2 eligible tracks to fit a palette, got {len(ids)} "
            f"(min_samples_fit={cfg.min_samples_fit}, "
            f"min_track_frames_fit={cfg.min_track_frames_fit})"
        )

    features = np.stack([observations[t].descriptor for t in ids])

    # 1. Provisional teams, 2. fitted rejection radius, 3. split off the anomalies.
    provisional = KMeans(n_clusters=2, n_init=20, random_state=_KMEANS_SEED).fit(features)
    radius = _fit_radius(features, provisional.cluster_centers_, cfg)
    distances = np.min(
        np.linalg.norm(features[:, None, :] - provisional.cluster_centers_[None, :, :], axis=2),
        axis=1,
    )
    is_outlier = distances > radius

    # 4. Refit the team centroids on the survivors, so an absorbed referee no longer
    #    drags a centroid.
    survivors = features[~is_outlier]
    if len(survivors) >= 2:
        team_fit = KMeans(n_clusters=2, n_init=20, random_state=_KMEANS_SEED).fit(survivors)
        team_centroids = team_fit.cluster_centers_
        labels, scored = team_fit.labels_, survivors
    else:  # degenerate clip: keep the provisional fit rather than fail outright
        team_centroids = provisional.cluster_centers_
        labels, scored = provisional.labels_, features

    # 5. Resolve the anomalies into goalkeepers and referees.
    outlier_ids = [t for t, flag in zip(ids, is_outlier, strict=True) if flag]
    resolution = _resolve(outlier_ids, observations, radius, cfg)

    # 6. Consistency pass. An outlier whose colour is still nearer a kit than its
    # assigned role is a noisy team member, not a referee; drop it and re-resolve.
    confirmed = [
        t
        for t in outlier_ids
        if _role_distance(observations[t].descriptor, t, resolution)
        <= float(np.min(np.linalg.norm(team_centroids - observations[t].descriptor, axis=1)))
    ]
    if len(confirmed) != len(outlier_ids):
        resolution = _resolve(confirmed, observations, radius, cfg)

    silhouette = (
        float(silhouette_score(scored, labels)) if len(set(labels.tolist())) > 1 else float("nan")
    )
    if not np.isnan(silhouette) and silhouette < cfg.min_silhouette:
        raise ValueError(
            f"team colours are not separable on this clip: silhouette {silhouette:.3f} "
            f"< min_silhouette {cfg.min_silhouette}. The two kits are too close in Lab "
            f"space for a two-team fit; a palette fitted here would be confidently wrong. "
            f"Lower teamid.min_silhouette only with a measurement that justifies it."
        )
    provenance = {
        "n_tracks": len(ids),
        "n_outliers": int(is_outlier.sum()),
        "fitted_track_ids": ids,
        "outlier_track_ids": outlier_ids,
        "role_track_ids": confirmed,
        "silhouette": silhouette,
        "reject_radius": float(radius),
    }
    return canonicalise_palette(
        TeamPalette(
            team_centroids=team_centroids,
            gk_centroids=resolution.gk_centroids,
            referee_centroid=resolution.referee_centroid,
            reject_radius=radius,
            provenance=provenance,
        )
    )


def _nearest_target(lab: np.ndarray, targets: np.ndarray, min_margin: float) -> int | None:
    """Index of the centroid a sample votes for, or ``None`` when the vote is ambiguous."""
    distances = np.linalg.norm(targets - np.asarray(lab, dtype=np.float64)[None, :], axis=1)
    order = np.argsort(distances)
    nearest = distances[order[0]]
    runner_up = distances[order[1]] if len(order) > 1 else np.inf
    if runner_up < nearest * min_margin:
        return None
    return int(order[0])


class TeamAssigner:
    """Accumulates per-track votes against a frozen palette.

    ``observe`` runs inline during tracking; ``consolidate`` produces the final
    per-track label once the clip is done.
    """

    def __init__(self, palette: TeamPalette, cfg: TeamIDConfig) -> None:
        self._palette = palette
        self._cfg = cfg
        self._votes: dict[int, Counter] = {}
        self._rejected: Counter = Counter()
        self._targets, self._labels = self.build_targets(palette)

    @staticmethod
    def build_targets(palette: TeamPalette) -> tuple[np.ndarray, list[tuple[Role, int]]]:
        """Stack every centroid into one array with a parallel (role, team) list."""
        centroids: list[np.ndarray] = []
        labels: list[tuple[Role, int]] = []
        for index, centroid in enumerate(palette.team_centroids):
            centroids.append(centroid)
            labels.append((Role.TEAM_A if index == 0 else Role.TEAM_B, index))
        for index, centroid in enumerate(palette.gk_centroids):
            centroids.append(centroid)
            labels.append((Role.GOALKEEPER, index))
        if palette.referee_centroid is not None:
            centroids.append(palette.referee_centroid)
            labels.append((Role.REFEREE, -1))
        return np.stack(centroids), labels

    def observe(self, track_id: int, lab: np.ndarray | None) -> None:
        """Record one frame's evidence for a track. ``None`` counts as a rejection."""
        if lab is None:
            self._rejected[track_id] += 1
            return

        winner = _nearest_target(lab, self._targets, self._cfg.min_margin)
        if winner is None:  # ambiguous between two centroids: not evidence
            self._rejected[track_id] += 1
            return
        self._votes.setdefault(track_id, Counter())[winner] += 1

    def label(self, track_id: int) -> TrackLabel:
        """Current best label for a track — ``unknown`` until the evidence gate clears."""
        votes = self._votes.get(track_id, Counter())
        total = sum(votes.values())
        rejected = self._rejected[track_id]
        if total < self._cfg.min_samples_assign:
            return TrackLabel(Role.UNKNOWN, -1, 0.0, total, rejected)

        winner, count = votes.most_common(1)[0]
        role, team = self._labels[winner]
        return TrackLabel(role, team, count / total, total, rejected)

    def consolidate(self, track_frames: dict[int, int] | None = None) -> dict[int, TrackLabel]:
        """Final label per track, over its whole life.

        Includes tracks that only ever produced rejected samples. Pass ``track_frames``
        (track id -> lifetime in frames) to apply the rare-role gate: a goalkeeper or
        referee label needs the same evidence as informing the palette fit, otherwise
        it becomes ``unknown``.
        """
        labels = {tid: self.label(tid) for tid in set(self._votes) | set(self._rejected)}
        if track_frames is None:
            return labels

        rare = (Role.GOALKEEPER, Role.REFEREE)
        for tid, label in labels.items():
            if label.role not in rare:
                continue
            eligible = (
                label.n_samples >= self._cfg.min_samples_fit
                and track_frames.get(tid, 0) >= self._cfg.min_track_frames_fit
            )
            if not eligible:
                labels[tid] = TrackLabel(Role.UNKNOWN, -1, 0.0, label.n_samples, label.n_rejected)
        return labels
