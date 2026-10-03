"""Ball tracking, Pass B: offline trajectory decisions.

Pure post-processing over the candidates ``ball.py`` collected: a min-cost path over
the whole candidate DAG picks at most one candidate per frame — the precision
mechanism, since a false positive loses to coasting — then gaps are interpolated and
airborne arcs flagged.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from football_tracker.homography.solve import project
from football_tracker.tracking.ball import BallCandidate

if TYPE_CHECKING:
    from football_tracker.config import BallConfig
    from football_tracker.homography.record import FrameCalibration

# Morphological closing of the airborne flag sequence: holes up to this many
# frames are filled before short islands are dropped.
_AIRBORNE_MAX_HOLE = 5
# The projected-speed signal only trusts detections at most this many frames apart.
# Across a larger gap the apparent speed is a gap-average that cannot distinguish a
# fast ground ball (far→near) from an airborne one, so it must not fire there.
_AIRBORNE_SPEED_MAX_DT = 3

_EYE3 = np.eye(3)


@dataclass(frozen=True)
class BallState:
    """Final per-frame ball output (crop-space pixels).

    ``xy`` is ``None`` only when ``source == "missing"``. ``confidence`` is the
    detector confidence for ``"detected"`` frames, else 0.0.
    """

    frame_idx: int
    xy: np.ndarray | None
    source: str  # "detected" | "interpolated" | "missing"
    airborne: bool
    confidence: float


def _as_h(affine_2x3: np.ndarray) -> np.ndarray:
    """Lift a 2x3 affine to a 3x3 homogeneous matrix."""
    h = np.eye(3)
    h[:2, :] = affine_2x3
    return h


def cumulative_warps(
    gmc_by_idx: dict[int, np.ndarray], t_min: int, t_max: int
) -> dict[int, np.ndarray]:
    """3x3 warp from frame-t_min space into each frame t's space, for t in [t_min, t_max].

    Frames missing from ``gmc_by_idx`` contribute identity, matching
    :class:`~football_tracker.tracking.gmc.PrecomputedCMC`.
    """
    warps = {t_min: _EYE3.copy()}
    current = _EYE3.copy()
    for t in range(t_min + 1, t_max + 1):
        affine = gmc_by_idx.get(t)
        if affine is not None:
            current = _as_h(affine) @ current
        warps[t] = current
    return warps


def solve_path(
    candidates: list[BallCandidate],
    gmc_by_idx: dict[int, np.ndarray],
    cfg: BallConfig,
) -> list[BallCandidate]:
    """Pick at most one candidate per frame: min-cost source→sink path over the DAG.

    Edges within ``max_gap`` frames carry a speed gate + speed cost measured in
    camera-compensated coordinates; longer gaps may still be bridged (the ball is
    genuinely lost — e.g. possession) with no motion constraint, paying only the
    per-frame miss cost. An empty path is allowed if every candidate costs more
    than missing.
    """
    if not candidates:
        return []
    cands = sorted(candidates, key=lambda c: c.frame_idx)
    t_min, t_max = cands[0].frame_idx, cands[-1].frame_idx

    # Map every candidate into the t_min reference space once, so per-pair distances
    # need one matmul.
    warps = cumulative_warps(gmc_by_idx, t_min, t_max)
    refs = np.empty((len(cands), 2))
    for i, c in enumerate(cands):
        w = warps[c.frame_idx]
        inv = np.linalg.inv(w)
        refs[i] = inv[:2, :2] @ c.xy + inv[:2, 2]

    node = [cfg.conf_weight * (1.0 - c.confidence) for c in cands]
    best = [float("inf")] * len(cands)
    parent = [-1] * len(cands)

    # Running minimum of (best[j] - miss_cost * t_j) over candidates beyond max_gap,
    # so arbitrarily long miss chains are O(1) instead of O(N) per candidate.
    long_min = float("inf")
    long_arg = -1
    long_ptr = 0

    for i, ci in enumerate(cands):
        ti = ci.frame_idx
        lin = warps[ti][:2, :2]
        # From source: pay for every skipped frame since the first candidate frame.
        best[i] = cfg.miss_cost * (ti - t_min) + node[i]

        # Motion edges
        j = i - 1
        while j >= 0 and ti - cands[j].frame_idx <= cfg.max_gap:
            gap = ti - cands[j].frame_idx
            if gap >= 1:
                diff = lin @ (refs[i] - refs[j])
                speed = float(np.linalg.norm(diff)) / gap
                if speed <= cfg.max_speed_px:
                    cost = (
                        best[j]
                        + cfg.miss_cost * (gap - 1)
                        + cfg.speed_weight * speed / cfg.max_speed_px
                        + node[i]
                    )
                    if cost < best[i]:
                        best[i] = cost
                        parent[i] = j
            j -= 1

        # Long edges (gap > max_gap): no motion constraint, misses only.
        while long_ptr < i and cands[long_ptr].frame_idx <= ti - cfg.max_gap - 1:
            value = best[long_ptr] - cfg.miss_cost * cands[long_ptr].frame_idx
            if value < long_min:
                long_min = value
                long_arg = long_ptr
            long_ptr += 1
        if long_arg >= 0:
            cost = long_min + cfg.miss_cost * (ti - 1) + node[i]
            if cost < best[i]:
                best[i] = cost
                parent[i] = long_arg

    # To sink: pay for skipped frames after the path's last candidate.
    totals = [best[i] + cfg.miss_cost * (t_max - c.frame_idx) for i, c in enumerate(cands)]
    end = int(np.argmin(totals))
    empty_cost = cfg.miss_cost * (t_max - t_min + 1)
    if totals[end] >= empty_cost:
        return []

    path: list[BallCandidate] = []
    i = end
    while i >= 0:
        path.append(cands[i])
        i = parent[i]
    path.reverse()
    return path


def interpolate_gaps(
    path: list[BallCandidate],
    frame_range: tuple[int, int],
    gmc_by_idx: dict[int, np.ndarray],
    cfg: BallConfig,
) -> list[BallState]:
    """Expand a solved path to one :class:`BallState` per frame in ``frame_range``.

    Gaps of at most ``max_interp_gap`` missing frames are filled linearly in
    GMC-stabilized coordinates (both endpoints warped into the gap frame's space,
    so camera pan does not bend the fill). Longer gaps — and frames before the
    first / after the last detection — stay ``"missing"``.
    ``frame_range`` is inclusive on both ends.
    """
    t_start, t_end = frame_range
    states: dict[int, BallState] = {
        t: BallState(frame_idx=t, xy=None, source="missing", airborne=False, confidence=0.0)
        for t in range(t_start, t_end + 1)
    }
    for cand in path:
        states[cand.frame_idx] = BallState(
            frame_idx=cand.frame_idx,
            xy=cand.xy.astype(np.float64),
            source="detected",
            airborne=False,
            confidence=cand.confidence,
        )
    ordered = sorted(path, key=lambda c: c.frame_idx)
    if len(ordered) < 2:
        return [states[t] for t in range(t_start, t_end + 1)]
    warps = cumulative_warps(gmc_by_idx, ordered[0].frame_idx, ordered[-1].frame_idx)
    for a, b in zip(ordered, ordered[1:], strict=False):
        gap = b.frame_idx - a.frame_idx - 1
        if gap == 0 or gap > cfg.max_interp_gap:
            continue
        # Both endpoints into the reference space once, then into each gap frame's space.
        a_ref = np.linalg.inv(warps[a.frame_idx]) @ np.append(a.xy.astype(np.float64), 1.0)
        b_ref = np.linalg.inv(warps[b.frame_idx]) @ np.append(b.xy.astype(np.float64), 1.0)
        for t in range(a.frame_idx + 1, b.frame_idx):
            pa = (warps[t] @ a_ref)[:2]
            pb = (warps[t] @ b_ref)[:2]
            alpha = (t - a.frame_idx) / (b.frame_idx - a.frame_idx)
            states[t] = BallState(
                frame_idx=t,
                xy=(1.0 - alpha) * pa + alpha * pb,
                source="interpolated",
                airborne=False,
                confidence=0.0,
            )
    return [states[t] for t in range(t_start, t_end + 1)]


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of the True runs in a boolean array."""
    runs: list[tuple[int, int]] = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def close_flags(flags: np.ndarray, max_hole: int, min_island: int) -> np.ndarray:
    """Morphological cleanup of a boolean flag sequence.

    Fills False holes of at most ``max_hole`` frames between True runs, then
    drops True islands shorter than ``min_island`` frames.
    """
    out = np.asarray(flags, dtype=bool).copy()
    runs = _true_runs(out)
    for (_, end_a), (start_b, _) in zip(runs, runs[1:], strict=False):
        if start_b - end_a <= max_hole:
            out[end_a:start_b] = True
    for start, end in _true_runs(out):
        if end - start < min_island:
            out[start:end] = False
    return out


def flag_airborne(
    states: list[BallState],
    gmc_by_idx: dict[int, np.ndarray],
    calib_by_idx: dict[int, FrameCalibration],
    cfg: BallConfig,
    fps: float,
) -> list[BallState]:
    """Mark airborne frames; returns new states with ``airborne`` set.

    Two signals on GMC-stabilized image coordinates, OR'd and then cleaned by
    :func:`close_flags`:

    1. **Parabola:** a least-squares quadratic over a sliding window shows
       gravity-signed curvature, a tight fit, and its apex inside the window.
    2. **Projected speed:** the flat-pitch projection sweeps faster than
       ``airborne_speed_ms`` for a sustained stretch.

    Only ``detected`` frames feed either signal: an interpolated fill is a straight
    line, so its apparent motion is an artifact, not evidence. The flag is conservative
    and fires rarely; downstream consumers gate on projected speed as well.
    """
    if not states:
        return []
    n = len(states)
    t0 = states[0].frame_idx
    warps = cumulative_warps(gmc_by_idx, t0, states[-1].frame_idx)

    # Stabilize only real observations into the first frame's coordinate space;
    # interpolated / missing positions do not count as evidence.
    stab = np.full((n, 2), np.nan)
    for i, s in enumerate(states):
        if s.xy is not None and s.source == "detected":
            inv = np.linalg.inv(warps[s.frame_idx])
            stab[i] = inv[:2, :2] @ s.xy + inv[:2, 2]
    valid = ~np.isnan(stab[:, 0])

    # Signal 1 — parabola fit per sliding window.
    parabola = np.zeros(n, dtype=bool)
    half = cfg.airborne_window // 2
    min_points = max(7, cfg.airborne_window // 2)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        idx = np.nonzero(valid[lo:hi])[0] + lo
        if len(idx) < min_points or not valid[i]:
            continue
        ts = idx.astype(np.float64) - i  # centered for conditioning
        ys = stab[idx, 1]
        a, b, c = np.polyfit(ts, ys, 2)
        if a < cfg.airborne_curvature_min:
            continue
        residual = ys - (a * ts * ts + b * ts + c)
        if float(np.sqrt(np.mean(residual**2))) > cfg.airborne_residual_px:
            continue
        apex = -b / (2.0 * a)
        if ts[0] <= apex <= ts[-1]:  # rise-then-fall inside the window
            parabola[i] = True

    # Signal 2 — sustained implausible projected ground speed, between consecutive
    # *detected* observations (interpolated gaps carry no speed evidence). A big gap
    # divides by its true dt, so a slow ball behind a long interpolation stays slow.
    obs_idx = np.nonzero(valid)[0]
    fast = np.zeros(n, dtype=bool)
    for a, b in zip(obs_idx, obs_idx[1:], strict=False):
        s1, s2 = states[a], states[b]
        dt = s2.frame_idx - s1.frame_idx
        c1, c2 = calib_by_idx.get(s1.frame_idx), calib_by_idx.get(s2.frame_idx)
        usable = (
            dt > 0
            and dt <= _AIRBORNE_SPEED_MAX_DT  # only closely-spaced detections carry speed evidence
            and c1 is not None
            and c2 is not None
            and c1.H is not None
            and c2.H is not None
            and not c1.low_confidence
            and not c2.low_confidence
        )
        if not usable:
            continue
        p1 = project(c1.H, s1.xy.reshape(1, 2))[0]
        p2 = project(c2.H, s2.xy.reshape(1, 2))[0]
        speed_ms = float(np.linalg.norm(p2 - p1)) * fps / dt
        if speed_ms > cfg.airborne_speed_ms:
            fast[a : b + 1] = True
    sustained = np.zeros(n, dtype=bool)
    for start, end in _true_runs(fast):
        if end - start >= cfg.airborne_min_frames:
            sustained[start:end] = True

    flags = close_flags(
        parabola | sustained, max_hole=_AIRBORNE_MAX_HOLE, min_island=cfg.airborne_min_frames
    )
    return [replace(s, airborne=bool(f)) for s, f in zip(states, flags, strict=True)]


def build_trajectory(
    candidates: list[BallCandidate],
    frame_range: tuple[int, int],
    stream: list[FrameCalibration],
    cfg: BallConfig,
    fps: float,
) -> list[BallState]:
    """Candidates → final per-frame trajectory: the whole of pass B."""
    gmc_by_idx = {c.frame_idx: c.gmc for c in stream}
    calib_by_idx = {c.frame_idx: c for c in stream}
    path = solve_path(candidates, gmc_by_idx, cfg)
    states = interpolate_gaps(path, frame_range, gmc_by_idx, cfg)
    return flag_airborne(states, gmc_by_idx, calib_by_idx, cfg, fps)
