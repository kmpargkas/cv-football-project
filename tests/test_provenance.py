"""Provenance guards: a calibration built from the wrong clip must be provable where it is built."""

from __future__ import annotations

import json

from football_tracker.homography.provenance import (
    anchor_fingerprint,
    check_anchor_coverage,
    provenance_block,
)


def test_coverage_accepts_anchors_inside_the_clip():
    coverage = check_anchor_coverage([0, 100, 200, 300], frame_count=400, max_gap_frames=150)
    assert coverage.ok
    assert coverage.fatal == []
    assert coverage.advisory == []


def test_an_anchor_past_the_end_of_the_clip_is_fatal():
    """Anchors labelled on a longer clip than this one: provable, so it can never be waived."""
    coverage = check_anchor_coverage([0, 400, 900], frame_count=732, max_gap_frames=150)
    assert len(coverage.fatal) == 1
    assert "900" in coverage.fatal[0]
    assert "732" in coverage.fatal[0]
    assert not coverage.ok


def test_a_wrong_clip_file_reports_nothing_but_the_mismatch():
    """Gap advisories against another clip's layout would suggest the wrong fix."""
    coverage = check_anchor_coverage([0, 400, 900], frame_count=732, max_gap_frames=150)
    assert coverage.advisory == []


def test_a_gap_wider_than_the_budget_is_advisory_not_fatal():
    """Wide gaps are a deliberate trade -- expensive, but sometimes chosen."""
    coverage = check_anchor_coverage([0, 400], frame_count=500, max_gap_frames=150)
    assert coverage.fatal == []
    assert any("400" in p and "gap" in p.lower() for p in coverage.advisory)


def test_an_unanchored_tail_is_advisory():
    """A clip running 600 frames past its last anchor is extrapolating, not propagating."""
    coverage = check_anchor_coverage([0, 100], frame_count=800, max_gap_frames=150)
    assert coverage.fatal == []
    assert any("tail" in p.lower() for p in coverage.advisory)


def test_an_unanchored_head_is_advisory():
    """Frames before the first anchor extrapolate just as the ones after the last do."""
    coverage = check_anchor_coverage([600, 700], frame_count=800, max_gap_frames=150)
    assert coverage.fatal == []
    assert any("head" in p.lower() for p in coverage.advisory)


def test_a_head_within_the_budget_is_not_flagged():
    """The boundary matches the tail's: wider than the budget, not equal to it."""
    at_budget = check_anchor_coverage([150, 300], frame_count=400, max_gap_frames=150)
    past_budget = check_anchor_coverage([151, 300], frame_count=400, max_gap_frames=150)
    assert not any("head" in p.lower() for p in at_budget.advisory)
    assert any("head" in p.lower() for p in past_budget.advisory)


def test_fewer_than_two_anchors_is_fatal():
    """Structural, not a budget call: there is nothing to propagate between."""
    coverage = check_anchor_coverage([0], frame_count=500, max_gap_frames=150)
    assert any("at least 2" in p for p in coverage.fatal)
    assert coverage.advisory == []


def test_fingerprint_is_stable_and_content_sensitive(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps({"frames": {"0": [[1, 2]]}}))
    b.write_text(json.dumps({"frames": {"0": [[9, 9]]}}))
    assert anchor_fingerprint(a) == anchor_fingerprint(a)
    assert anchor_fingerprint(a) != anchor_fingerprint(b)
    assert len(anchor_fingerprint(a)) == 12


def test_provenance_block_records_both_inputs(tmp_path):
    anchors = tmp_path / "pitch_points_clip.json"
    anchors.write_text(json.dumps({"frames": {"0": [[1, 2]]}}))
    block = provenance_block(
        video=tmp_path / "clip.mp4",
        anchors_path=anchors,
        frame_count=1390,
        anchor_frames=[0, 198, 396],
    )
    assert block["video"].endswith("clip.mp4")
    assert block["anchors"].endswith("pitch_points_clip.json")
    assert block["anchors_sha256_12"] == anchor_fingerprint(anchors)
    assert block["clip_frame_count"] == 1390
    assert block["anchor_frames"] == [0, 198, 396]
    assert block["anchor_gaps"] == [198, 198]
