"""The pitch every other module measures in: dimensions, the 32-vertex layout, and lines.

Coordinates are in meters, with (0, 0) at the top-left. The vertex layout and edge lists
are vendored from Roboflow's ``sports`` repo.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

# 1-based vertex index pairs of straight line segments between keypoints,
# vendored verbatim from sports.configs.soccer.SoccerPitchConfiguration.edges.
_EDGES: tuple[tuple[int, int], ...] = (
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (7, 8),
    (10, 11),
    (11, 12),
    (12, 13),
    (14, 15),
    (15, 16),
    (16, 17),
    (18, 19),
    (19, 20),
    (20, 21),
    (23, 24),
    (25, 26),
    (26, 27),
    (27, 28),
    (28, 29),
    (29, 30),
    (1, 14),
    (2, 10),
    (3, 7),
    (4, 8),
    (5, 13),
    (6, 17),
    (14, 25),
    (18, 26),
    (23, 27),
    (24, 28),
    (21, 29),
    (17, 30),
)


# Vertices that can not be localised (0-based).
# 94% of clicks are RANSAC outliers.
PHANTOM_VERTICES: tuple[int, ...] = (10, 11, 18, 19)


@dataclass(frozen=True)
class PitchSpec:
    """Regulation pitch dimensions in meters."""

    length: float = 105.0
    width: float = 68.0
    penalty_box_length: float = 16.5
    penalty_box_width: float = 40.32
    goal_box_length: float = 5.5
    goal_box_width: float = 18.32
    centre_circle_radius: float = 9.15
    penalty_spot_distance: float = 11.0

    @cached_property
    def keypoints(self) -> np.ndarray:
        """(32, 2) pitch coordinates matching the Roboflow model's output order."""
        length, width = self.length, self.width
        pb_l, pb_w = self.penalty_box_length, self.penalty_box_width
        gb_l, gb_w = self.goal_box_length, self.goal_box_width
        circle_r = self.centre_circle_radius
        spot = self.penalty_spot_distance
        vertices = np.array(
            [
                (0, 0),  # 1  left corner, top touchline
                (0, (width - pb_w) / 2),  # 2
                (0, (width - gb_w) / 2),  # 3
                (0, (width + gb_w) / 2),  # 4
                (0, (width + pb_w) / 2),  # 5
                (0, width),  # 6  left corner, bottom touchline
                (gb_l, (width - gb_w) / 2),  # 7
                (gb_l, (width + gb_w) / 2),  # 8
                (spot, width / 2),  # 9  left penalty spot
                (pb_l, (width - pb_w) / 2),  # 10
                (pb_l, (width - gb_w) / 2),  # 11
                (pb_l, (width + gb_w) / 2),  # 12
                (pb_l, (width + pb_w) / 2),  # 13
                (length / 2, 0),  # 14 halfway, top touchline
                (length / 2, width / 2 - circle_r),  # 15
                (length / 2, width / 2 + circle_r),  # 16
                (length / 2, width),  # 17 halfway, bottom touchline
                (length - pb_l, (width - pb_w) / 2),  # 18
                (length - pb_l, (width - gb_w) / 2),  # 19
                (length - pb_l, (width + gb_w) / 2),  # 20
                (length - pb_l, (width + pb_w) / 2),  # 21
                (length - spot, width / 2),  # 22 right penalty spot
                (length - gb_l, (width - gb_w) / 2),  # 23
                (length - gb_l, (width + gb_w) / 2),  # 24
                (length, 0),  # 25 right corner, top touchline
                (length, (width - pb_w) / 2),  # 26
                (length, (width - gb_w) / 2),  # 27
                (length, (width + gb_w) / 2),  # 28
                (length, (width + pb_w) / 2),  # 29
                (length, width),  # 30 right corner, bottom touchline
                (length / 2 - circle_r, width / 2),  # 31 circle, left tangent
                (length / 2 + circle_r, width / 2),  # 32 circle, right tangent
            ],
            dtype=np.float64,
        )
        vertices.flags.writeable = False
        return vertices

    @cached_property
    def boundary(self) -> np.ndarray:
        """(4, 2) pitch corners, clockwise from (0, 0)."""
        corners = np.array(
            [(0, 0), (self.length, 0), (self.length, self.width), (0, self.width)], dtype=np.float64
        )
        corners.flags.writeable = False
        return corners

    @cached_property
    def edges(self) -> tuple[tuple[int, int], ...]:
        """0-based keypoint index pairs of straight pitch-line segments."""
        return tuple((a - 1, b - 1) for a, b in _EDGES)

    def line_segments(self, circle_points: int = 48) -> list[np.ndarray]:
        """All drawable pitch lines as (N, 2) polylines in pitch meters.

        Straight segments from :attr:`edges`, plus the centre circle and the
        two penalty arcs sampled as polylines (for overlay rendering).
        """
        kp = self.keypoints
        segments: list[np.ndarray] = [np.stack([kp[a], kp[b]]) for a, b in self.edges]
        segments.append(self._circle(circle_points))
        segments.extend(self._penalty_arcs(circle_points))
        return segments

    def _circle(self, n: int) -> np.ndarray:
        theta = np.linspace(0.0, 2.0 * np.pi, n + 1)
        centre = np.array([self.length / 2, self.width / 2])
        radius = self.centre_circle_radius
        return centre + radius * np.stack([np.cos(theta), np.sin(theta)], axis=1)

    def _penalty_arcs(self, n: int) -> list[np.ndarray]:
        """The parts of the two penalty arcs outside their penalty boxes.

        Empty when the box reaches further from the spot than the arc's radius: the arc
        is then wholly inside the box and none of it is drawn.
        """
        radius = self.centre_circle_radius
        reach = self.penalty_box_length - self.penalty_spot_distance
        if radius <= 0.0 or reach >= radius:
            return []
        # Arc extends beyond the box edge: half-angle from the box-edge chord.
        half = np.arccos(max(reach / radius, -1.0))
        theta = np.linspace(-half, half, max(n // 4, 8))
        arcs = []
        for spot_x, sign in (
            (self.penalty_spot_distance, 1.0),
            (self.length - self.penalty_spot_distance, -1.0),
        ):
            centre = np.array([spot_x, self.width / 2])
            arcs.append(centre + radius * np.stack([sign * np.cos(theta), np.sin(theta)], axis=1))
        return arcs


def default_control_points(spec: PitchSpec) -> np.ndarray:
    """(6, 2) well-spread pitch points: the 4 corners plus the halfway-line ends."""
    halfway = [(spec.length / 2, 0.0), (spec.length / 2, spec.width)]
    return np.vstack([spec.boundary, halfway])
