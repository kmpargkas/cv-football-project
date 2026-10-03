"""Renders of the tracking outputs.

Debug: :class:`TrackAnnotator` draws a box and ``#<id> <class>`` label per track,
coloured by track id or by role, and :class:`BallAnnotator` draws the ball with a
fading trail coloured by source. Deliverable: :class:`KitAnnotator` draws an outline
ellipse and a soft glow at each player's feet in the measured kit colour, and
:func:`draw_ball_marker` a black ring round the ball, with an id pill under each
ellipse when ``labels`` are given -- no boxes.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np
import supervision as sv

from football_tracker.detection.interface import ObjectClass
from football_tracker.reid.roles import Role
from football_tracker.reid.teamid import KitColours, TrackLabel
from football_tracker.tracking.ball_trajectory import BallState
from football_tracker.tracking.tracker import TrackedObjects

# BGR per ball-state source: how the position was obtained must be visible.
_BALL_COLORS = {
    "detected": (80, 220, 80),  # green
    "interpolated": (60, 220, 240),  # yellow
}

# BGR per role, for the TeamID visual pass. Deliberately *not* the kit colours: the
# point of this render is spotting mistakes, and a white team drawn in white on a
# bright pitch hides exactly the errors we are looking for. These five are mutually
# distinct and all legible against grass.
_ROLE_ORDER = (Role.TEAM_A, Role.TEAM_B, Role.GOALKEEPER, Role.REFEREE, Role.UNKNOWN)
ROLE_COLORS = {
    Role.TEAM_A: (0, 140, 255),  # orange
    Role.TEAM_B: (255, 150, 0),  # blue
    Role.GOALKEEPER: (0, 255, 255),  # yellow
    Role.REFEREE: (255, 0, 255),  # magenta
    Role.UNKNOWN: (150, 150, 150),  # grey
}

BALL_MARKER_BGR = (0, 0, 0)
GLOW_ALPHA = 0.55
_ID_FONT = cv2.FONT_HERSHEY_SIMPLEX


class TrackAnnotator:
    """Draws boxes + labels for a frame's tracks.

    Colours by track id by default, so an id switch shows as a colour jump. Pass ``teams``
    (from ``teams.csv``) to colour by role instead — then the question being checked is
    whether a referee ever wears a team colour, not whether an id survived.
    """

    def __init__(
        self,
        thickness: int = 2,
        text_scale: float = 0.5,
        teams: dict[int, TrackLabel] | None = None,
    ) -> None:
        self._teams = teams
        lookup = sv.ColorLookup.CLASS if teams is not None else sv.ColorLookup.TRACK
        palette = (
            sv.ColorPalette([sv.Color(*reversed(ROLE_COLORS[r])) for r in _ROLE_ORDER])
            if teams is not None
            else sv.ColorPalette.DEFAULT
        )
        self._box = sv.RoundBoxAnnotator(thickness=thickness, color=palette, color_lookup=lookup)
        self._label = sv.LabelAnnotator(
            text_scale=text_scale, text_thickness=1, color=palette, color_lookup=lookup
        )

    def annotate(self, frame: np.ndarray, tracks: TrackedObjects) -> np.ndarray:
        """Return a copy of ``frame`` with ``tracks`` drawn on it (input never mutated)."""
        scene = frame.copy()
        if len(tracks) == 0:
            return scene

        if self._teams is not None:
            roles = [
                (self._teams[int(i)].role if int(i) in self._teams else Role.UNKNOWN)
                for i in tracks.id
            ]
            class_id = np.array([_ROLE_ORDER.index(r) for r in roles], dtype=int)
            labels = [f"#{int(i)} {r.value}" for i, r in zip(tracks.id, roles, strict=True)]
        else:
            class_id = tracks.class_id.astype(int)
            labels = [
                f"#{int(i)} {ObjectClass(int(c)).name.lower()}"
                for i, c in zip(tracks.id, tracks.class_id, strict=True)
            ]

        sv_det = sv.Detections(
            xyxy=tracks.xyxy.astype(np.float32).reshape(-1, 4),
            class_id=class_id,
            tracker_id=tracks.id.astype(int),
        )
        scene = self._box.annotate(scene, sv_det)
        scene = self._label.annotate(scene, sv_det, labels=labels)
        return scene


def draw_ground_glow(
    scene: np.ndarray, xyxy: np.ndarray, color: tuple[int, int, int], alpha: float = GLOW_ALPHA
) -> None:
    """Soft pool of light at the feet, fading up the body to nothing at the head; in place."""
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    w, h = x2 - x1, y2 - y1
    if w < 2 or h < 2:
        return
    blur = max(3, int(h / 6) | 1)
    frame_h, frame_w = scene.shape[:2]
    rx1, ry1 = max(0, int(x1) - blur), max(0, int(y1) - blur)
    rx2, ry2 = min(frame_w, int(x2) + blur + 1), min(frame_h, int(y2) + blur + 1)
    if rx2 <= rx1 or ry2 <= ry1:
        return
    cx, feet_y, half_w = round((x1 + x2) / 2 - rx1), round(y2 - ry1), w / 2
    shape = (ry2 - ry1, rx2 - rx1)

    body = np.zeros(shape, dtype=np.uint8)
    axes = (max(1, round(0.9 * half_w)), max(1, round(h / 2)))
    cv2.ellipse(body, (cx, round(feet_y - h / 2)), axes, 0.0, 0.0, 360.0, 255, -1)
    rows = (np.arange(shape[0], dtype=np.float32) - (y1 - ry1)) / h
    wash = body.astype(np.float32) / 255 * (np.clip(rows, 0, 1) ** 2)[:, None] * 0.6

    pool = np.zeros(shape, dtype=np.uint8)
    axes = (max(1, round(1.2 * half_w)), max(1, round(0.5 * half_w)))
    cv2.ellipse(pool, (cx, feet_y), axes, 0.0, 0.0, 360.0, 255, -1)

    mask = cv2.GaussianBlur(np.maximum(wash, pool.astype(np.float32) / 255), (blur, blur), 0)
    weight = (alpha * mask)[:, :, None]
    roi = scene[ry1:ry2, rx1:rx2].astype(np.float32)
    lit = roi * (1 - weight) + np.array(color, dtype=np.float32) * weight
    scene[ry1:ry2, rx1:rx2] = lit.round().clip(0, 255).astype(np.uint8)


def draw_ground_ellipse(
    scene: np.ndarray, xyxy: np.ndarray, color: tuple[int, int, int], thickness: int = 2
) -> None:
    """Outline arc across the bottom of the box, drawn in place."""
    x1, _, x2, y2 = (float(v) for v in xyxy)
    half_w = max(1, round((x2 - x1) / 2))
    centre = (round((x1 + x2) / 2), round(y2))
    axes = (half_w, max(1, round(0.35 * half_w)))
    cv2.ellipse(scene, centre, axes, 0.0, -45.0, 235.0, color, thickness, cv2.LINE_AA)


def draw_id_pill(
    scene: np.ndarray, xyxy: np.ndarray, color: tuple[int, int, int], text: str
) -> None:
    """Kit-coloured pill with ``text`` just below the arc, sized from the box; in place."""
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    half_w = max(1, round((x2 - x1) / 2))
    scale = float(np.clip((y2 - y1) / 100, 0.7, 1.0))
    thickness = 1 if scale < 0.85 else 2
    (tw, th), baseline = cv2.getTextSize(text, _ID_FONT, scale, thickness)
    pad = max(3, round(4 * scale))
    cx, top = round((x1 + x2) / 2), round(y2 + 0.35 * half_w) + 4
    left, right, bottom = cx - tw // 2 - pad, cx + tw // 2 + pad, top + th + baseline + pad
    radius = (bottom - top) // 2
    cv2.rectangle(scene, (left, top), (right, bottom), color, -1, cv2.LINE_AA)
    cv2.circle(scene, (left, top + radius), radius, color, -1, cv2.LINE_AA)
    cv2.circle(scene, (right, top + radius), radius, color, -1, cv2.LINE_AA)
    b, g, r = color
    ink = (0, 0, 0) if 0.114 * b + 0.587 * g + 0.299 * r > 150 else (255, 255, 255)
    origin = (cx - tw // 2, top + pad + th)
    cv2.putText(scene, text, origin, _ID_FONT, scale, ink, thickness, cv2.LINE_AA)


class KitAnnotator:
    """A glow and an outline ellipse at each track's feet in its kit colour.

    ``labels`` adds a pill under the arc with the given text per track id; tracks
    without an entry get none.
    """

    def __init__(
        self,
        teams: dict[int, TrackLabel],
        colours: KitColours,
        thickness: int = 2,
        labels: dict[int, str] | None = None,
    ) -> None:
        self._teams = teams
        self._colours = colours
        self._thickness = thickness
        self._labels = labels or {}

    def annotate(self, frame: np.ndarray, tracks: TrackedObjects) -> np.ndarray:
        """Return a copy of ``frame`` with ``tracks`` drawn on it (input never mutated)."""
        scene = frame.copy()
        colors = [self._colours.for_label(self._teams.get(int(i))) for i in tracks.id]
        # Every glow first, so no arc is washed out by a neighbour's glow.
        for xyxy, color in zip(tracks.xyxy, colors, strict=True):
            draw_ground_glow(scene, xyxy, color)
        for xyxy, color in zip(tracks.xyxy, colors, strict=True):
            draw_ground_ellipse(scene, xyxy, color, self._thickness)
        for xyxy, color, track_id in zip(tracks.xyxy, colors, tracks.id, strict=True):
            text = self._labels.get(int(track_id))
            if text is not None:
                draw_id_pill(scene, xyxy, color, text)
        return scene


def draw_ball_marker(
    frame: np.ndarray, xy: np.ndarray, radius: int = 10, thickness: int = 2
) -> np.ndarray:
    """Return a copy of ``frame`` with a black outline ellipse round the ball."""
    scene = frame.copy()
    centre = (round(float(xy[0])), round(float(xy[1])))
    axes = (radius, max(1, round(0.6 * radius)))
    cv2.ellipse(scene, centre, axes, 0.0, 0.0, 360.0, BALL_MARKER_BGR, thickness, cv2.LINE_AA)
    return scene


class BallAnnotator:
    """Draws the per-frame ball state + a fading trail (stateful across frames)."""

    def __init__(self, trail_length: int = 40, radius: int = 8) -> None:
        self._radius = radius
        # (xy, source) of recent frames with a position; oldest fades out first.
        self._trail: deque[tuple[np.ndarray, str]] = deque(maxlen=trail_length)

    def annotate(
        self,
        frame: np.ndarray,
        state: BallState,
        prediction: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return a copy of ``frame`` with the ball drawn (input never mutated)."""
        scene = frame.copy()
        if state.xy is not None:
            self._trail.append((state.xy.copy(), state.source))
        elif self._trail:
            # No position this frame: age the trail out instead of freezing it on
            # screen. A deque only evicts on append, so without this the last 40
            # circles persist for the whole missing stretch — drawing a ball path
            # exactly where the tracker is saying it has no ball.
            self._trail.popleft()
        for age, (xy, source) in enumerate(self._trail):
            fade = (age + 1) / len(self._trail)
            color = tuple(int(c * fade) for c in _BALL_COLORS[source])
            cv2.circle(scene, self._to_int(xy), max(2, int(self._radius * fade)), color, -1)
        if state.xy is not None:
            center = self._to_int(state.xy)
            color = _BALL_COLORS[state.source]
            cv2.circle(scene, center, self._radius, color, -1)
            if state.airborne:
                cv2.circle(scene, center, self._radius * 2, (255, 255, 255), 2)
                cv2.putText(
                    scene,
                    "AIR",
                    (center[0] + self._radius * 2 + 4, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
        if prediction is not None:
            px, py = self._to_int(prediction)
            size = self._radius
            cv2.line(scene, (px - size, py), (px + size, py), (255, 255, 255), 1)
            cv2.line(scene, (px, py - size), (px, py + size), (255, 255, 255), 1)
        return scene

    @staticmethod
    def _to_int(xy: np.ndarray) -> tuple[int, int]:
        return int(round(float(xy[0]))), int(round(float(xy[1])))
