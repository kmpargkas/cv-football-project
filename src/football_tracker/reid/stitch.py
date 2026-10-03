"""Offline tracklet stitching: rejoin the fragments one person was split into.

The online tracker decides frame by frame, forwards only. Seeing the whole clip at
once adds three constraints:

1. **Temporal separation** — two tracklets alive at the same time are two people. A
   few frames of overlap are tolerated as a handover seam; anything longer is
   forbidden.

2. **Role agreement** — a label vetoes a link only when both sides are confident and
   disagree; a weak label adds a cost instead.

3. **Roster cardinality** — reported, never enforced: merging can only reduce the
   number of simultaneous identities, so it is a diagnostic for under-merging.

The cost is pitch-space geometry: reachability at sprint speed, then velocity
extrapolation from one tracklet's tail to the next one's head. Links are solved as a
min-cost assignment, so each tracklet takes at most one successor and one
predecessor, and every chain is one identity.

Pure: arrays and mapping in, mapping out. No video, no model, no filesystem.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from football_tracker.config import StitchConfig
from football_tracker.reid.roles import Role
from football_tracker.reid.teamid import TrackLabel
from football_tracker.tracking.tracker import TrackedObjects


@dataclass(frozen=True)
class Tracklet:
    """One output track, summarised at the two ends that can be joined to something."""

    track_id: int
    role: Role  # only equal roles may link
    first_frame: int
    last_frame: int
    head_xy: np.ndarray  # (2,) pitch metres at first_frame
    tail_xy: np.ndarray  # (2,) pitch metres at last_frame
    tail_velocity: np.ndarray  # (2,) metres/second, averaged over the last N frames
    n_frames: int
    role_confidence: float = 1.0  # vote share behind `role`; a weak label cannot veto


def frames_for(seconds: float, fps: float) -> int:
    """Duration in seconds -> whole frames, never below 1."""
    return max(1, round(seconds * fps))


def build_tracklets(
    positions_by_frame: Mapping[int, Mapping[int, np.ndarray]],
    labels: Mapping[int, TrackLabel],
    fps: float,
    velocity_window_s: float = 0.08,
) -> list[Tracklet]:
    """Fold per-frame pitch positions into one :class:`Tracklet` per track.

    Tracks with no calibrated position anywhere are dropped: the cost model is
    geometric, so they could only ever be linked on a guess.
    """
    frames_by_track: dict[int, list[int]] = {}
    for frame, per_track in positions_by_frame.items():
        for track_id in per_track:
            frames_by_track.setdefault(int(track_id), []).append(int(frame))

    tracklets: list[Tracklet] = []
    for track_id, frames in frames_by_track.items():
        frames.sort()
        points = np.stack([positions_by_frame[f][track_id] for f in frames]).astype(np.float64)
        label = labels.get(track_id)
        tracklets.append(
            Tracklet(
                track_id=track_id,
                role=Role.UNKNOWN if label is None else label.role,
                first_frame=frames[0],
                last_frame=frames[-1],
                head_xy=points[0],
                tail_xy=points[-1],
                tail_velocity=_velocity(frames, points, fps, frames_for(velocity_window_s, fps)),
                n_frames=len(frames),
                role_confidence=0.0 if label is None else float(label.confidence),
            )
        )
    return sorted(tracklets, key=lambda t: (t.first_frame, t.track_id))


def _velocity(frames: list[int], points: np.ndarray, fps: float, window: int) -> np.ndarray:
    """Metres/second over the last ``window`` frames, or zero when there is one sample."""
    if len(frames) < 2:
        return np.zeros(2)
    tail = points[-window:]
    span = frames[-1] - frames[-len(tail)]
    if span <= 0:
        return np.zeros(2)
    return (tail[-1] - tail[0]) * fps / span


def link_cost(a: Tracklet, b: Tracklet, fps: float, cfg: StitchConfig) -> float | None:
    """Cost of continuing ``a`` with ``b``, or ``None`` when the link is forbidden.

    Forbidden links stay out of the cost matrix entirely, so the solver cannot reach
    for one when nothing better is available.
    """
    role_penalty = 0.0
    if a.role is not b.role:
        if min(a.role_confidence, b.role_confidence) >= cfg.role_trust_conf:
            return None  # two confident, disagreeing labels: different people
        role_penalty = cfg.w_role_mismatch

    tolerance = frames_for(cfg.overlap_tolerance_s, fps)
    if b.first_frame < a.last_frame - tolerance:
        return None  # sustained overlap: two people, or an upstream NMS duplicate

    dt = max(1, b.first_frame - a.last_frame) / fps
    if dt > cfg.max_gap_s:
        return None

    # A fragment too short to have a trustworthy velocity may still be absorbed, but
    # only across a gap small enough that geometry alone decides it.
    if (
        min(a.n_frames, b.n_frames) < frames_for(cfg.min_tracklet_s, fps)
        and dt > cfg.short_fragment_max_gap_s
    ):
        return None

    reach = cfg.v_max_ms * dt + cfg.pos_sigma_m
    if float(np.linalg.norm(b.head_xy - a.tail_xy)) > reach:
        return None  # unreachable at a sprint

    # Extrapolate the exit velocity across the gap: a player who ran off in a direction
    # and reappears along it is a much better match than one who reappears behind them.
    predicted = a.tail_xy + a.tail_velocity * dt
    error = float(np.linalg.norm(b.head_xy - predicted))
    return cfg.w_pos * (error / reach) + cfg.w_time * (dt / cfg.max_gap_s) + role_penalty


def solve_links(tracklets: Sequence[Tracklet], fps: float, cfg: StitchConfig) -> dict[int, int]:
    """Track id -> merged identity id, by min-cost assignment over allowed links.

    Each tracklet takes at most one successor and at most one predecessor, so the
    solution is a set of chains and every chain is one person. Links costing more than
    ``max_link_cost`` are dropped after solving rather than before: the assignment is
    global, so a link that looks acceptable alone may still be the wrong global choice.
    Every tracklet takes part; the length gate lives in :func:`link_cost`.
    """
    from scipy.optimize import linear_sum_assignment

    remap = {t.track_id: t.track_id for t in tracklets}
    if len(tracklets) < 2:
        return remap

    n = len(tracklets)
    big = 1e6
    cost = np.full((n, n), big)
    for i, a in enumerate(tracklets):
        for j, b in enumerate(tracklets):
            if i == j:
                continue
            c = link_cost(a, b, fps, cfg)
            if c is not None:
                cost[i, j] = c

    rows, cols = linear_sum_assignment(cost)
    successor = {
        tracklets[i].track_id: tracklets[j].track_id
        for i, j in zip(rows, cols, strict=True)
        if cost[i, j] <= cfg.max_link_cost
    }

    # Walk each chain to its head, so every member of a chain maps to one identity.
    predecessors = {v: k for k, v in successor.items()}
    for tracklet in tracklets:
        root = tracklet.track_id
        seen = {root}
        while root in predecessors and predecessors[root] not in seen:
            root = predecessors[root]
            seen.add(root)
        remap[tracklet.track_id] = root
    return remap


def roster_overflow(
    tracklets: Sequence[Tracklet], remap: Mapping[int, int], roster_outfield: int = 11
) -> dict[Role, int]:
    """Role -> how many identities beyond the roster are ever simultaneously alive.

    A diagnostic for under merging; cardinality never forbids a link.
    """
    by_role: dict[Role, dict[int, list[tuple[int, int]]]] = {}
    for t in tracklets:
        ident = remap.get(t.track_id, t.track_id)
        by_role.setdefault(t.role, {}).setdefault(ident, []).append((t.first_frame, t.last_frame))

    overflow: dict[Role, int] = {}
    for role, identities in by_role.items():
        if role is Role.UNKNOWN:
            continue
        events: list[tuple[int, int]] = []
        for spans in identities.values():
            events.append((min(f for f, _ in spans), 1))
            events.append((max(last for _, last in spans) + 1, -1))
        live = peak = 0
        for _, delta in sorted(events):
            live += delta
            peak = max(peak, live)
        if peak > roster_outfield:
            overflow[role] = peak - roster_outfield
    return overflow


def apply_remap(
    tracks_by_frame: Mapping[int, TrackedObjects], remap: Mapping[int, int]
) -> dict[int, TrackedObjects]:
    """Rewrite track ids across every frame, keeping one box per identity per frame.

    Overlap tolerance in :func:`link_cost` lets two tracklets that share a handover seam
    be joined, which would otherwise leave two rows with one id on those frames. The
    surviving row is the one whose original id *is* the merge target; failing that, the
    most confident.
    """
    out: dict[int, TrackedObjects] = {}
    for frame, tracks in tracks_by_frame.items():
        new_ids = np.array([remap.get(int(t), int(t)) for t in tracks.id], dtype=np.int64)
        keep: dict[int, int] = {}
        for row, (old, new) in enumerate(zip(tracks.id, new_ids, strict=True)):
            best = keep.get(int(new))
            if best is None:
                keep[int(new)] = row
                continue
            incumbent_is_root = int(tracks.id[best]) == int(new)
            challenger_is_root = int(old) == int(new)
            if challenger_is_root != incumbent_is_root:
                wins = challenger_is_root  # the surviving id's own box is the real one
            else:
                wins = bool(tracks.confidence[row] > tracks.confidence[best])
            if wins:
                keep[int(new)] = row
        rows = sorted(keep.values())
        out[frame] = TrackedObjects(
            xyxy=tracks.xyxy[rows],
            id=new_ids[rows],
            confidence=tracks.confidence[rows],
            class_id=tracks.class_id[rows],
        )
    return out
