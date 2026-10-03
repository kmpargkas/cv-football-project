"""Team and role assignment from kit colour, plus offline tracklet stitching.

One torso colour per detection; a palette fitted once per clip, with goalkeepers and
referees split off as colour outliers before the two teams are fitted; then a
per-track vote against that frozen palette. Stitching runs afterwards over the whole
clip and rejoins the fragments one person was split into, on pitch geometry and role
agreement.
"""

from football_tracker.reid.appearance import torso_descriptor
from football_tracker.reid.roles import Role
from football_tracker.reid.stitch import (
    Tracklet,
    apply_remap,
    build_tracklets,
    roster_overflow,
    solve_links,
)
from football_tracker.reid.teamid import (
    TeamAssigner,
    TeamPalette,
    TrackLabel,
    TrackObservation,
    build_observations,
    canonicalise_palette,
    describe_team_order,
    fit_palette,
)

__all__ = [
    # The colour feature and the categories it resolves into.
    "torso_descriptor",
    "Role",
    # Fitting a clip's palette, then labelling its tracks.
    "TrackObservation",
    "build_observations",
    "fit_palette",
    "canonicalise_palette",
    "describe_team_order",
    "TeamPalette",
    "TeamAssigner",
    "TrackLabel",
    # Offline stitching.
    "Tracklet",
    "build_tracklets",
    "solve_links",
    "roster_overflow",
    "apply_remap",
]
