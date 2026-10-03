"""Radar geometry and drawing (pure: arrays in, arrays out)."""

import numpy as np
import pytest

from football_tracker.homography.pitch import PitchSpec
from football_tracker.projection.radar import (
    GLASS_BGR,
    LINE_BGR,
    Dot,
    Minimap,
    draw_minimap,
    draw_possession_bar,
    draw_radar_frame,
    hex_to_bgr,
    minimap_geometry,
    pitch_to_px,
    render_pitch,
)

SPEC = PitchSpec()


def test_hex_to_bgr():
    assert hex_to_bgr("#d62828") == (40, 40, 214)
    assert hex_to_bgr("#ffffff") == (255, 255, 255)


def test_pitch_to_px_corners_and_centre():
    pts = np.array([[0.0, 0.0], [105.0, 68.0], [52.5, 34.0]])
    px = pitch_to_px(pts, scale=10.0, margin=40)
    np.testing.assert_array_equal(px[0], [40, 40])
    np.testing.assert_array_equal(px[1], [1090, 720])
    np.testing.assert_array_equal(px[2], [565, 380])


def test_render_pitch_dimensions_and_lines():
    img = render_pitch(SPEC, scale=10.0, margin=40)
    assert img.shape == (760, 1130, 3)  # 68*10 + 2*40, 105*10 + 2*40
    assert img.dtype == np.uint8
    # the halfway line is white at the pitch centre column (cv2 line rasterisation
    # may paint 564-565 or 565-566 for a 2px-wide line centred on column 565)
    assert (img[400, 564:567] > 200).any()
    # the margin is background, not line
    assert (img[10, 10] < 200).any()


def test_draw_radar_frame_paints_dot_and_preserves_background():
    bg = render_pitch(SPEC, scale=10.0, margin=40)
    before = bg.copy()
    dot = Dot(xy_m=np.array([52.5, 34.0]), color=(40, 40, 214), radius_px=6, outline=False)
    out = draw_radar_frame(bg, [dot], banner="test", scale=10.0, margin=40)
    np.testing.assert_array_equal(bg, before)  # input untouched
    assert tuple(out[380, 565]) == (40, 40, 214)  # dot centre painted


def test_render_pitch_boxes_circle_and_spots():
    # scale=10.0, margin=40, row 380 == the pitch centreline (width/2 == 34.0 m).
    # Windows are a +-1 column neighbourhood around the target, same tolerance as
    # the halfway-line test, to absorb cv2 line-rasterisation choice.
    img = render_pitch(SPEC, scale=10.0, margin=40)

    # Left penalty box far edge: x = 16.5 m -> column 40 + 165 = 205.
    assert (img[380, 204:207] > 200).any()
    # Right penalty box far edge: x = 105 - 16.5 = 88.5 m -> column 40 + 885 = 925.
    # This is the assertion that actually catches a sign error in the right-hand
    # box's `direction=-1.0` mirroring: get that wrong and the line either lands
    # off the 1130-px-wide image (column 1255) or is never drawn at all.
    assert (img[380, 924:927] > 200).any()

    # Centre circle left extreme: x = 52.5 - 9.15 = 43.35 m -> column 40 + 433.5 ~= 471.
    assert (img[380, 470:473] > 200).any()
    # Centre circle right extreme: x = 52.5 + 9.15 = 61.65 m -> column 40 + 616.5 ~= 658.
    assert (img[380, 657:660] > 200).any()

    # Left penalty spot (filled): x = 11 m -> column 40 + 110 = 150.
    assert (img[380, 149:152] > 200).any()
    # Right penalty spot (filled): x = 105 - 11 = 94 m -> column 40 + 940 = 980.
    assert (img[380, 979:982] > 200).any()


def test_draw_radar_frame_outline_ring():
    bg = render_pitch(SPEC, scale=10.0, margin=40)
    dot = Dot(xy_m=np.array([52.5, 34.0]), color=(40, 40, 214), radius_px=6, outline=True)
    out = draw_radar_frame(bg, [dot], banner="", scale=10.0, margin=40)
    assert tuple(out[380, 565]) == (40, 40, 214)  # centre stays the fill colour
    # a ring pixel just outside the fill radius (radius_px + 1 = 7 px to the right)
    assert tuple(out[380, 572]) == (245, 245, 245)


def test_draw_radar_frame_banner():
    bg = render_pitch(SPEC, scale=10.0, margin=40)
    top_margin = np.s_[0:40, 0:400]
    out_with_banner = draw_radar_frame(bg, [], banner="test", scale=10.0, margin=40)
    out_without_banner = draw_radar_frame(bg, [], banner="", scale=10.0, margin=40)
    assert not np.array_equal(out_with_banner[top_margin], bg[top_margin])
    np.testing.assert_array_equal(out_without_banner[top_margin], bg[top_margin])


def test_draw_radar_frame_hollow_dot_is_a_ring_not_a_disc():
    """A `filled=False` dot must not paint its centre, or it reads as a confident
    dot that happens to be bigger."""
    bg = render_pitch(SPEC, scale=10.0, margin=40)
    dot = Dot(
        xy_m=np.array([52.5, 34.0]),
        color=(40, 40, 214),
        radius_px=8,
        outline=False,
        filled=False,
    )
    out = draw_radar_frame(bg, [dot], banner="", scale=10.0, margin=40)
    assert tuple(out[380, 565]) == tuple(bg[380, 565])  # centre untouched
    assert tuple(out[380, 573]) == (40, 40, 214)  # rim painted at radius 8


def test_dots_are_filled_by_default():
    bg = render_pitch(SPEC, scale=10.0, margin=40)
    dot = Dot(xy_m=np.array([52.5, 34.0]), color=(40, 40, 214), radius_px=6, outline=False)
    out = draw_radar_frame(bg, [dot], banner="", scale=10.0, margin=40)
    assert tuple(out[380, 565]) == (40, 40, 214)


# --- glass minimap ---------------------------------------------------------------------


def test_render_pitch_background_is_configurable():
    img = render_pitch(SPEC, scale=10.0, margin=40, background=GLASS_BGR)
    assert tuple(img[10, 10]) == GLASS_BGR


def test_minimap_geometry_sits_bottom_centre_and_matches_render_pitch():
    g = minimap_geometry((1920, 1080), SPEC)
    assert isinstance(g, Minimap)
    assert g.w == 480  # 25 % of 1920
    assert g.x == (1920 - g.w) // 2
    assert g.y + g.h == 1080 - round(1080 * 0.03)
    assert render_pitch(SPEC, g.scale, g.margin).shape == (g.h, g.w, 3)


def test_minimap_geometry_refuses_a_frame_it_cannot_fit():
    with pytest.raises(ValueError):
        minimap_geometry((400, 60), SPEC)


def test_draw_minimap_blends_the_pitch_and_keeps_dots_opaque():
    scene = np.full((1080, 1920, 3), 200, dtype=np.uint8)
    g = minimap_geometry((1920, 1080), SPEC)
    dot = Dot(xy_m=np.array([30.0, 20.0]), color=(40, 40, 214), radius_px=4, outline=False)
    out = draw_minimap(scene, [dot], SPEC, g)
    assert out.shape == scene.shape
    assert (scene == 200).all()  # input untouched
    assert (out[: g.y] == 200).all()  # nothing above the panel
    # a plain glass pixel inside the margin is the 45 % blend of grey over the frame
    assert abs(int(out[g.y + 3, g.x + 3, 0]) - round(200 * 0.55 + 40 * 0.45)) <= 1
    # the dot centre is exactly the dot colour
    cx, cy = pitch_to_px(dot.xy_m, g.scale, g.margin)[0]
    assert tuple(out[g.y + cy, g.x + cx]) == (40, 40, 214)


def test_draw_minimap_keeps_markings_and_border_opaque():
    scene = np.zeros((1080, 1920, 3), dtype=np.uint8)
    g = minimap_geometry((1920, 1080), SPEC)
    out = draw_minimap(scene, [], SPEC, g)
    assert tuple(out[g.y, g.x]) == LINE_BGR  # border corner
    cx, cy = pitch_to_px(np.array([52.5, 34.0]), g.scale, g.margin)[0]
    row = out[g.y + cy, g.x + cx - 1 : g.x + cx + 2]
    assert (row == LINE_BGR).all(axis=1).any()  # halfway line, full strength


def test_draw_minimap_ball_is_black_with_a_white_ring():
    scene = np.full((1080, 1920, 3), 200, dtype=np.uint8)
    g = minimap_geometry((1920, 1080), SPEC)
    ball = Dot(xy_m=np.array([30.0, 20.0]), color=(0, 0, 0), radius_px=3, outline=True)
    out = draw_minimap(scene, [ball], SPEC, g)
    cx, cy = pitch_to_px(ball.xy_m, g.scale, g.margin)[0]
    assert tuple(out[g.y + cy, g.x + cx]) == (0, 0, 0)
    assert tuple(out[g.y + cy, g.x + cx + 4]) == LINE_BGR


# --- possession bar --------------------------------------------------------------------

_A, _B = (40, 40, 214), (230, 230, 230)


def test_possession_bar_splits_at_the_share_in_each_teams_colour():
    scene = np.full((1080, 1920, 3), 120, dtype=np.uint8)
    g = minimap_geometry((1920, 1080), SPEC)
    out = draw_possession_bar(scene, g, 0.7, (_A, _B))
    assert (scene == 120).all()  # input untouched
    rows = out[: g.y]
    hit = np.where((rows == _A).all(axis=2).any(axis=1))[0]
    assert len(hit) > 0
    row = rows[hit[len(hit) // 2]]
    a_cols = np.where((row == _A).all(axis=1))[0]
    b_cols = np.where((row == _B).all(axis=1))[0]
    assert a_cols.max() < b_cols.min()  # team A on the left
    split = (a_cols.max() + 1 - g.x) / g.w
    assert abs(split - 0.7) < 0.02


def test_possession_bar_sits_just_above_the_minimap_and_spans_its_width():
    scene = np.full((1080, 1920, 3), 120, dtype=np.uint8)
    g = minimap_geometry((1920, 1080), SPEC)
    out = draw_possession_bar(scene, g, 0.5, (_A, _B))
    changed = np.where((out != scene).any(axis=2))
    assert changed[0].max() < g.y  # never overlaps the minimap
    assert g.y - changed[0].max() <= 12  # but sits right above it
    assert changed[1].min() == g.x and changed[1].max() == g.x + g.w - 1
    assert changed[0].max() - changed[0].min() <= 16  # slim


def test_possession_bar_before_any_possession_is_empty_glass():
    scene = np.full((1080, 1920, 3), 120, dtype=np.uint8)
    g = minimap_geometry((1920, 1080), SPEC)
    out = draw_possession_bar(scene, g, None, (_A, _B))
    assert not (out == _A).all(axis=2).any()
    assert not (out == _B).all(axis=2).any()
    assert (out[: g.y] != 120).any()  # the empty bar is still drawn
