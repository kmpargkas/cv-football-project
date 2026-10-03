"""Round trips for the TeamID artifacts: teams.csv and palette.json."""

from __future__ import annotations

import numpy as np
import pytest

from football_tracker.reid.roles import Role
from football_tracker.reid.teamid import TeamPalette, TrackLabel
from football_tracker.tracking.artifacts import (
    read_palette,
    read_teams_csv,
    write_palette,
    write_teams_csv,
)

LABELS = {
    3: TrackLabel(Role.TEAM_A, 0, 1.0, 41, 0),
    7: TrackLabel(Role.TEAM_B, 1, 0.78, 1277, 7),
    16: TrackLabel(Role.GOALKEEPER, -1, 1.0, 1262, 14),
    18: TrackLabel(Role.REFEREE, -1, 0.95, 1115, 167),
    44: TrackLabel(Role.UNKNOWN, -1, 0.0, 0, 8),
}


def _palette(with_referee: bool = True) -> TeamPalette:
    return TeamPalette(
        team_centroids=np.array([[234.2, 123.1, 134.7], [141.4, 167.4, 150.0]]),
        gk_centroids=np.array([[226.0, 115.0, 200.0]]),
        referee_centroid=np.array([85.0, 126.0, 137.0]) if with_referee else None,
        reject_radius=20.0,
        provenance={"n_tracks": 29, "silhouette": 0.917, "fitted_track_ids": [3, 7, 16]},
    )


def test_teams_csv_round_trips(tmp_path):
    path = tmp_path / "teams.csv"
    write_teams_csv(path, LABELS)
    restored = read_teams_csv(path)
    assert set(restored) == set(LABELS)
    for tid, want in LABELS.items():
        got = restored[tid]
        assert got.role is want.role
        assert got.team == want.team
        assert got.confidence == pytest.approx(want.confidence, abs=1e-3)
        assert got.n_samples == want.n_samples
        assert got.n_rejected == want.n_rejected


def test_teams_csv_has_a_header_and_one_row_per_track(tmp_path):
    path = tmp_path / "teams.csv"
    write_teams_csv(path, LABELS)
    lines = path.read_text().strip().splitlines()
    assert lines[0].startswith("track_id,")
    assert len(lines) == len(LABELS) + 1


def test_teams_csv_rows_are_sorted_by_track_id(tmp_path):
    """Deterministic output: a re-run must produce a byte-identical file."""
    path = tmp_path / "teams.csv"
    write_teams_csv(path, LABELS)
    ids = [int(line.split(",")[0]) for line in path.read_text().strip().splitlines()[1:]]
    assert ids == sorted(ids)


def test_unknown_tracks_survive_the_round_trip(tmp_path):
    """`unknown` is a first-class value, not an omission."""
    path = tmp_path / "teams.csv"
    write_teams_csv(path, LABELS)
    assert read_teams_csv(path)[44].role is Role.UNKNOWN


def test_empty_labels_write_a_header_only_file(tmp_path):
    path = tmp_path / "teams.csv"
    write_teams_csv(path, {})
    assert path.read_text().strip().splitlines() == [
        "track_id,team,role,confidence,n_samples,n_rejected"
    ]
    assert read_teams_csv(path) == {}


def test_writer_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "nested" / "deeper" / "teams.csv"
    write_teams_csv(path, LABELS)
    assert path.exists()


def test_palette_round_trips(tmp_path):
    path = tmp_path / "palette.json"
    original = _palette()
    write_palette(path, original)
    restored = read_palette(path)
    assert np.allclose(restored.team_centroids, original.team_centroids)
    assert np.allclose(restored.gk_centroids, original.gk_centroids)
    assert np.allclose(restored.referee_centroid, original.referee_centroid)
    assert restored.reject_radius == original.reject_radius
    assert restored.provenance["silhouette"] == pytest.approx(0.917)


def test_palette_round_trips_without_a_referee(tmp_path):
    path = tmp_path / "palette.json"
    write_palette(path, _palette(with_referee=False))
    assert read_palette(path).referee_centroid is None


def test_palette_with_no_goalkeepers_round_trips(tmp_path):
    path = tmp_path / "palette.json"
    original = TeamPalette(
        team_centroids=np.array([[234.0, 123.0, 135.0], [141.0, 167.0, 150.0]]),
        gk_centroids=np.zeros((0, 3)),
        referee_centroid=None,
        reject_radius=25.5,
        provenance={},
    )
    write_palette(path, original)
    restored = read_palette(path)
    assert restored.gk_centroids.shape == (0, 3)
