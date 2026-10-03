"""Torso colour descriptor — the TeamID feature.

A grass-suppressed CIELAB median over the torso band separates two kits at
silhouette 0.789 per track, and goalkeepers and referees sit 47-68 Lab units from
the nearest team centroid against an in-team scatter of 5-20, so a 3-vector is
enough. Kits that differ by pattern rather than mean colour (red vs red-and-black
stripes) would need a histogram instead.

Pure: arrays in, array out. No model, no filesystem, no video.
"""

from __future__ import annotations

import cv2
import numpy as np

from football_tracker.config import TeamIDConfig


def _band_bounds(
    xyxy: np.ndarray, shape: tuple[int, int], cfg: TeamIDConfig
) -> tuple[int, int, int, int] | None:
    """Pixel bounds of the torso band, clipped to the frame, or ``None`` if degenerate."""
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    height, width = y2 - y1, x2 - x1
    if height < cfg.min_box_h_px:
        return None

    frame_h, frame_w = shape
    half = cfg.torso_width / 2.0
    y0 = max(0, int(round(y1 + cfg.torso_top * height)))
    y3 = min(frame_h, int(round(y1 + cfg.torso_bottom * height)))
    x0 = max(0, int(round(x1 + width * (0.5 - half))))
    x3 = min(frame_w, int(round(x1 + width * (0.5 + half))))
    if y3 - y0 < 3 or x3 - x0 < 3:
        return None
    return y0, y3, x0, x3


def torso_descriptor(
    frame_bgr: np.ndarray, xyxy: np.ndarray, cfg: TeamIDConfig
) -> np.ndarray | None:
    """Median torso colour for one detection, or ``None`` when a quality gate fails.

    Returns a ``(3,)`` float32 CIELAB vector in OpenCV's 0-255 ranges.

    Rejects rather than down-weights. Clean tracks show a within-track Lab spread of
    5-15 while contaminated ones reach 30-52 — the contamination being an overlapping
    player or background intruding into the band. A per-track median absorbs a few
    bad samples; the palette fit does not.

    ``frame_bgr`` must be the same letterbox crop the rest of the pipeline uses, so
    that ``xyxy`` and the frame share one pixel space.
    """
    bounds = _band_bounds(xyxy, frame_bgr.shape[:2], cfg)
    if bounds is None:
        return None
    y0, y1, x0, x1 = bounds

    band = frame_bgr[y0:y1, x0:x1]
    lab = cv2.cvtColor(band, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)

    grass = (
        (hsv[:, 0] > cfg.grass_hue_lo)
        & (hsv[:, 0] < cfg.grass_hue_hi)
        & (hsv[:, 1] > cfg.grass_sat_min)
    )
    extreme = (lab[:, 0] < cfg.shadow_l_min) | (lab[:, 0] > cfg.highlight_l_max)
    keep = ~(grass | extreme)

    if float(keep.mean()) < cfg.min_kept_fraction:
        return None
    return np.median(lab[keep], axis=0).astype(np.float32)
