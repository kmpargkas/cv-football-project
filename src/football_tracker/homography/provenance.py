"""Bind a calibration stream to the clip and anchor set that produced it.

A ``calibration.npz`` is read by every stage that follows with wrong anchors
being indistinguishable from a correct ones until some metric collapses far from the cause.
The fingerprint and layout recorded here make that mismatch provable at the point it happens.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AnchorCoverage:
    """Problems with an anchor layout, split by whether they can be waived.

    ``fatal`` problems admit no override: the layout is structurally unusable, or the
    labels provably belong to another clip.``advisory`` problems are budget calls an
    operator may accept.
    """

    fatal: list[str] = field(default_factory=list)
    advisory: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.fatal and not self.advisory


def check_anchor_coverage(
    anchor_frames: list[int],
    frame_count: int,
    max_gap_frames: int,
) -> AnchorCoverage:
    """Inspect an anchor layout against the clip it claims to describe."""
    frames = sorted(anchor_frames)

    # Structural, not a budget call: with fewer than two anchors there is nothing
    # to propagate between, so there is no layout to assess.
    if len(frames) < 2:
        return AnchorCoverage(
            fatal=[f"need at least 2 anchors to propagate between, got {len(frames)}"]
        )
    beyond = [f for f in frames if f >= frame_count]
    if beyond:
        return AnchorCoverage(
            fatal=[
                f"anchor frame(s) {beyond} are at or past the clip's frame_count "
                f"({frame_count}) -- these anchors were labelled on a different clip"
            ]
        )

    advisory: list[str] = []
    for a, b in zip(frames, frames[1:], strict=False):
        if b - a > max_gap_frames:
            advisory.append(
                f"gap of {b - a} frames between anchors {a} and {b} exceeds the "
                f"{max_gap_frames}-frame propagation budget"
            )

    # Both ends are one-way: fusion brackets a frame between two anchors, so outside
    # the anchored span there is only one estimate and it extrapolates.
    if frames[0] > max_gap_frames:
        advisory.append(
            f"unanchored head of {frames[0]} frames before the first anchor: "
            f"the stream extrapolates rather than propagates there"
        )

    tail = frame_count - 1 - frames[-1]
    if tail > max_gap_frames:
        advisory.append(
            f"unanchored tail of {tail} frames after the last anchor ({frames[-1]}): "
            f"the stream extrapolates rather than propagates there"
        )

    return AnchorCoverage(advisory=advisory)


def anchor_fingerprint(anchors_path: str | Path) -> str:
    """First 12 hex chars of the anchors file's sha256 -- enough to detect a swap."""
    data = Path(anchors_path).read_bytes()
    return hashlib.sha256(data).hexdigest()[:12]


def provenance_block(
    video: str | Path,
    anchors_path: str | Path,
    frame_count: int,
    anchor_frames: list[int],
) -> dict:
    """The block merged into ``calibration_metrics.json``.

    Records both inputs and the derived anchor layout, so a future reader can
    tell which clip and which labels a stream came from without trusting a
    directory name.
    """
    frames = sorted(anchor_frames)
    return {
        "video": str(video),
        "anchors": str(anchors_path),
        "anchors_sha256_12": anchor_fingerprint(anchors_path),
        "clip_frame_count": frame_count,
        "anchor_frames": frames,
        "anchor_gaps": [b - a for a, b in zip(frames, frames[1:], strict=False)],
    }
