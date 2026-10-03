"""2D pitch radar: metre→pixel mapping and cv2 drawing.

Lines and dots share one metre→pixel mapping, so nothing has to be reverse
engineered from a plotting library's figure margins.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from football_tracker.homography.pitch import PitchSpec

GRASS_BGR = (60, 110, 45)
LINE_BGR = (245, 245, 245)
BANNER_BGR = (20, 20, 20)
GLASS_BGR = (40, 40, 40)
GLASS_ALPHA = 0.45


def hex_to_bgr(color: str) -> tuple[int, int, int]:
    """'#rrggbb' → cv2's (B, G, R)."""
    c = color.lstrip("#")
    return int(c[4:6], 16), int(c[2:4], 16), int(c[0:2], 16)


def pitch_to_px(xy_m: np.ndarray, scale: float, margin: int) -> np.ndarray:
    """(N, 2) pitch metres → (N, 2) integer radar pixels."""
    xy = np.asarray(xy_m, dtype=np.float64).reshape(-1, 2)
    return np.rint(xy * scale + margin).astype(int)


def _pt(x_m: float, y_m: float, scale: float, margin: int) -> tuple[int, int]:
    x, y = pitch_to_px(np.array([x_m, y_m]), scale, margin)[0]
    return int(x), int(y)


def render_pitch(
    spec: PitchSpec,
    scale: float,
    margin: int,
    background: tuple[int, int, int] = GRASS_BGR,
) -> np.ndarray:
    """Pitch markings in white on ``background``, drawn from the same spec the H maps to."""
    w = round(spec.length * scale) + 2 * margin
    h = round(spec.width * scale) + 2 * margin
    img = np.full((h, w, 3), background, dtype=np.uint8)
    length, width = spec.length, spec.width
    thick = max(1, round(scale / 6))

    def rect(x0: float, y0: float, x1: float, y1: float) -> None:
        cv2.rectangle(img, _pt(x0, y0, scale, margin), _pt(x1, y1, scale, margin), LINE_BGR, thick)

    rect(0, 0, length, width)  # touchlines + goal lines
    cv2.line(
        img,
        _pt(length / 2, 0, scale, margin),
        _pt(length / 2, width, scale, margin),
        LINE_BGR,
        thick,
    )
    cv2.circle(
        img,
        _pt(length / 2, width / 2, scale, margin),
        round(spec.centre_circle_radius * scale),
        LINE_BGR,
        thick,
    )
    pb_y0 = (width - spec.penalty_box_width) / 2
    gb_y0 = (width - spec.goal_box_width) / 2
    for x0, direction in ((0.0, 1.0), (length, -1.0)):
        rect(x0, pb_y0, x0 + direction * spec.penalty_box_length, pb_y0 + spec.penalty_box_width)
        rect(x0, gb_y0, x0 + direction * spec.goal_box_length, gb_y0 + spec.goal_box_width)
        cv2.circle(
            img,
            _pt(x0 + direction * spec.penalty_spot_distance, width / 2, scale, margin),
            max(2, thick),
            LINE_BGR,
            -1,
        )
    return img


@dataclass(frozen=True)
class Dot:
    """One rendered marker: a player, official, or the ball."""

    xy_m: np.ndarray  # (2,)
    color: tuple[int, int, int]  # BGR
    radius_px: int
    outline: bool  # white ring (the ball)
    # False draws the rim only. Used for the ball's uncertainty marker, where a
    # filled disc would claim a position the projection cannot support -- see
    # projection/uncertainty.py. Defaults True so every other dot is unchanged.
    filled: bool = True


def _paint_dot(img: np.ndarray, dot: Dot, scale: float, margin: int) -> None:
    x, y = pitch_to_px(dot.xy_m, scale, margin)[0]
    cv2.circle(img, (x, y), dot.radius_px, dot.color, -1 if dot.filled else 2)
    if dot.outline:
        cv2.circle(img, (x, y), dot.radius_px + 1, LINE_BGR, 1)


def draw_radar_frame(
    background: np.ndarray,
    dots: list[Dot],
    banner: str,
    scale: float,
    margin: int,
) -> np.ndarray:
    """Copy the background, paint the dots, and write the banner line."""
    img = background.copy()
    for dot in dots:
        _paint_dot(img, dot, scale, margin)
    if banner:
        origin = (margin, max(24, margin - 12))
        for color, thickness in ((BANNER_BGR, 3), (LINE_BGR, 1)):  # dark stroke under white text
            cv2.putText(
                img, banner, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness, cv2.LINE_AA
            )
    return img


@dataclass(frozen=True)
class Minimap:
    """The glass pitch's rectangle on a video frame and the metre→pixel mapping inside it."""

    x: int
    y: int
    w: int
    h: int
    scale: float
    margin: int


def minimap_geometry(
    frame_wh: tuple[int, int],
    spec: PitchSpec,
    width_frac: float = 0.25,
    bottom_margin_frac: float = 0.03,
) -> Minimap:
    """Bottom-centre panel, ``width_frac`` of the frame wide, floating above the bottom edge."""
    frame_w, frame_h = frame_wh
    panel_w = round(frame_w * width_frac)
    margin = max(2, round(panel_w * 0.025))
    scale = (panel_w - 2 * margin) / spec.length
    # Same rounding as render_pitch, so the panel it draws is exactly this size.
    w = round(spec.length * scale) + 2 * margin
    h = round(spec.width * scale) + 2 * margin
    x = (frame_w - w) // 2
    y = frame_h - h - round(frame_h * bottom_margin_frac)
    if x < 0 or y < 0:
        raise ValueError(f"a {w}x{h} minimap does not fit a {frame_w}x{frame_h} frame")
    return Minimap(x=x, y=y, w=w, h=h, scale=scale, margin=margin)


def draw_minimap(
    scene: np.ndarray,
    dots: list[Dot],
    spec: PitchSpec,
    geom: Minimap,
    alpha: float = GLASS_ALPHA,
) -> np.ndarray:
    """Copy of ``scene`` with a translucent pitch at ``geom``; markings, border and dots opaque."""
    img = scene.copy()
    panel = render_pitch(spec, geom.scale, geom.margin, background=GLASS_BGR)
    roi = img[geom.y : geom.y + geom.h, geom.x : geom.x + geom.w]
    glass = cv2.addWeighted(roi, 1 - alpha, panel, alpha, 0)
    glass[(panel == LINE_BGR).all(axis=2)] = LINE_BGR
    cv2.rectangle(glass, (0, 0), (geom.w - 1, geom.h - 1), LINE_BGR, 1)
    for dot in dots:
        _paint_dot(glass, dot, geom.scale, geom.margin)
    img[geom.y : geom.y + geom.h, geom.x : geom.x + geom.w] = glass
    return img


def draw_possession_bar(
    scene: np.ndarray,
    geom: Minimap,
    share_a: float | None,
    colors: tuple[tuple[int, int, int], tuple[int, int, int]],
    alpha: float = GLASS_ALPHA,
) -> np.ndarray:
    """Copy of ``scene`` with a slim bar just above the minimap: team A's share on the left.

    ``share_a`` is team A's fraction of the frames either team held the ball; ``None``
    draws the empty glass bar.
    """
    img = scene.copy()
    h = max(6, round(geom.w * 0.012))
    gap = max(4, round(h * 0.75))
    y0 = geom.y - gap - h
    if y0 < 0:
        return img
    roi = img[y0 : y0 + h, geom.x : geom.x + geom.w]
    bar = cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, GLASS_BGR), alpha, 0)
    if share_a is not None:
        split = round(float(np.clip(share_a, 0.0, 1.0)) * geom.w)
        bar[:, :split] = colors[0]
        bar[:, split:] = colors[1]
    cv2.rectangle(bar, (0, 0), (geom.w - 1, h - 1), LINE_BGR, 1)
    img[y0 : y0 + h, geom.x : geom.x + geom.w] = bar
    return img
