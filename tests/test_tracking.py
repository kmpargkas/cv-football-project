"""Player-tracking glue: GMC injection and off-field rejection."""

from __future__ import annotations

import numpy as np
import pytest

from football_tracker.detection.interface import Detections, ObjectClass
from football_tracker.homography.pitch import PitchSpec
from football_tracker.homography.record import FrameCalibration
from football_tracker.tracking.gmc import PrecomputedCMC, off_field_mask

IDENTITY_2X3 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _calib(frame_idx: int, H, source: str, low_confidence: bool) -> FrameCalibration:
    return FrameCalibration(
        frame_idx=frame_idx,
        H=H,
        source=source,
        low_confidence=low_confidence,
        gmc=IDENTITY_2X3,
    )


class TestPrecomputedCMC:
    def test_returns_stored_affine_for_current_frame(self):
        a5 = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 4.0]])
        cmc = PrecomputedCMC({5: a5})
        cmc.set_frame(5)
        np.testing.assert_allclose(cmc.apply(None, None), a5)

    def test_identity_for_missing_frame(self):
        cmc = PrecomputedCMC({})
        cmc.set_frame(99)
        np.testing.assert_allclose(cmc.apply(None, None), IDENTITY_2X3)

    def test_from_stream_indexes_by_frame_idx(self):
        gmc = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]])
        stream = [
            FrameCalibration(frame_idx=7, H=None, source="none", low_confidence=True, gmc=gmc)
        ]
        cmc = PrecomputedCMC.from_stream(stream)
        cmc.set_frame(7)
        np.testing.assert_allclose(cmc.apply(None, None), gmc)

    def test_apply_returns_2x3_float(self):
        cmc = PrecomputedCMC({0: IDENTITY_2X3})
        cmc.set_frame(0)
        warp = cmc.apply(np.zeros((4, 4, 3), np.uint8), np.zeros((0, 4)))
        assert warp.shape == (2, 3)
        assert np.issubdtype(warp.dtype, np.floating)


class TestOffFieldMask:
    def test_keeps_on_pitch_drops_off_pitch(self):
        # H = identity → foot points map to themselves in pitch meters.
        # foot of [5,5,15,20] = (10, 20) inside 105x68; foot of [195,...,200] = (200,200) outside.
        dets = Detections(
            xyxy=[[5, 5, 15, 20], [195, 195, 205, 200]],
            confidence=[0.9, 0.9],
            class_id=[ObjectClass.PLAYER, ObjectClass.PLAYER],
        )
        calib = _calib(0, np.eye(3), "solved", low_confidence=False)
        mask = off_field_mask(dets, calib, PitchSpec(), margin_m=1.0)
        assert mask.tolist() == [True, False]

    def test_passthrough_when_H_none(self):
        dets = Detections(xyxy=[[195, 195, 205, 200]], confidence=[0.9], class_id=[2])
        calib = _calib(0, None, "none", low_confidence=True)
        mask = off_field_mask(dets, calib, PitchSpec(), margin_m=1.0)
        assert mask.tolist() == [True]

    def test_passthrough_when_low_confidence(self):
        dets = Detections(xyxy=[[195, 195, 205, 200]], confidence=[0.9], class_id=[2])
        calib = _calib(0, np.eye(3), "propagated", low_confidence=True)
        mask = off_field_mask(dets, calib, PitchSpec(), margin_m=1.0)
        assert mask.tolist() == [True]

    def test_passthrough_when_calib_missing(self):
        dets = Detections(xyxy=[[195, 195, 205, 200]], confidence=[0.9], class_id=[2])
        mask = off_field_mask(dets, None, PitchSpec(), margin_m=1.0)
        assert mask.tolist() == [True]

    def test_empty_dets(self):
        calib = _calib(0, np.eye(3), "solved", low_confidence=False)
        mask = off_field_mask(Detections.empty(), calib, PitchSpec())
        assert mask.tolist() == []


class TestPlayerTracker:
    """Wrapper glue over a real BoxMOT backend (motion-only, no ReID weights)."""

    def _sequence_calib(self, n: int) -> list[FrameCalibration]:
        return [_calib(i, np.eye(3), "solved", low_confidence=False) for i in range(n)]

    def test_drops_ball_and_off_field_keeps_stable_id(self):
        pytest.importorskip("boxmot")
        from football_tracker.config import TrackingConfig
        from football_tracker.tracking.tracker import PlayerTracker

        stream = self._sequence_calib(5)
        cfg = TrackingConfig()
        tracker = PlayerTracker.from_config(cfg, stream, PitchSpec(), fps=30.0)

        ids_seen: set[int] = set()
        for idx in range(5):
            # A player drifting slowly (on pitch), plus a ball and an off-field box
            # that must be dropped before association.
            x = 40 + idx  # pitch-x ~40 m, well inside
            dets = Detections(
                xyxy=[
                    [x, 30, x + 8, 50],  # player, foot ~(x+4, 50) on pitch
                    [x, 30, x + 4, 40],  # ball — must be dropped
                    [300, 300, 320, 340],  # off-field — must be dropped
                ],
                confidence=[0.9, 0.9, 0.9],
                class_id=[ObjectClass.PLAYER, ObjectClass.BALL, ObjectClass.PLAYER],
            )
            tracked = tracker.update(idx, np.zeros((400, 400, 3), np.uint8), dets)
            # Only the on-pitch player should ever produce a track; never the ball.
            assert all(int(c) != int(ObjectClass.BALL) for c in tracked.class_id)
            ids_seen.update(int(i) for i in tracked.id)

        # A single consistently-tracked player → exactly one identity.
        assert len(ids_seen) == 1


class TestClassConditionalMask:
    """Assistant referees work the touchline, so the margin is widened for officials
    only -- the same band also holds warming-up substitutes."""

    @staticmethod
    def _dets(classes):
        # H = identity, so foot points are pitch metres. Foot of each box is (x2+x1)/2,
        # y2 -> all at y = 70, i.e. 2 m outside the 68 m touchline.
        return Detections(
            xyxy=[[10, 60, 20, 70]] * len(classes),
            confidence=[0.9] * len(classes),
            class_id=list(classes),
        )

    def _calib(self):
        return _calib(0, np.eye(3), "solved", low_confidence=False)

    def test_referee_two_metres_out_is_kept_while_a_player_is_dropped(self):
        dets = self._dets([ObjectClass.REFEREE, ObjectClass.PLAYER])
        mask = off_field_mask(
            dets,
            self._calib(),
            PitchSpec(),
            margin_m=1.0,
            class_margins={int(ObjectClass.REFEREE): 3.0},
        )
        assert mask.tolist() == [True, False]

    def test_a_substitute_warming_up_is_still_excluded(self):
        """The whole point: widening for officials must not admit players."""
        dets = self._dets([ObjectClass.PLAYER, ObjectClass.GOALKEEPER])
        mask = off_field_mask(
            dets,
            self._calib(),
            PitchSpec(),
            margin_m=1.0,
            class_margins={int(ObjectClass.REFEREE): 3.0},
        )
        assert mask.tolist() == [False, False]

    def test_referee_beyond_its_own_margin_is_still_dropped(self):
        dets = Detections(
            xyxy=[[10, 60, 20, 80]],  # foot at y = 80, i.e. 12 m outside
            confidence=[0.9],
            class_id=[ObjectClass.REFEREE],
        )
        mask = off_field_mask(
            dets,
            self._calib(),
            PitchSpec(),
            margin_m=1.0,
            class_margins={int(ObjectClass.REFEREE): 3.0},
        )
        assert mask.tolist() == [False]

    def test_omitting_class_margins_reproduces_the_scalar_behaviour(self):
        dets = self._dets([ObjectClass.REFEREE, ObjectClass.PLAYER])
        scalar = off_field_mask(dets, self._calib(), PitchSpec(), margin_m=1.0)
        explicit = off_field_mask(
            dets, self._calib(), PitchSpec(), margin_m=1.0, class_margins=None
        )
        assert scalar.tolist() == explicit.tolist() == [False, False]

    def test_a_class_without_an_override_uses_the_default_margin(self):
        dets = self._dets([ObjectClass.GOALKEEPER])
        mask = off_field_mask(
            dets,
            self._calib(),
            PitchSpec(),
            margin_m=3.0,
            class_margins={int(ObjectClass.REFEREE): 3.0},
        )
        assert mask.tolist() == [True]

    def test_passthrough_on_bad_calibration_ignores_class_margins(self):
        dets = self._dets([ObjectClass.PLAYER, ObjectClass.REFEREE])
        mask = off_field_mask(
            dets,
            _calib(0, None, "none", low_confidence=True),
            PitchSpec(),
            margin_m=1.0,
            class_margins={int(ObjectClass.REFEREE): 3.0},
        )
        assert mask.tolist() == [True, True]

    def test_empty_detections(self):
        dets = Detections(xyxy=np.zeros((0, 4)), confidence=[], class_id=[])
        mask = off_field_mask(
            dets,
            self._calib(),
            PitchSpec(),
            margin_m=1.0,
            class_margins={int(ObjectClass.REFEREE): 3.0},
        )
        assert mask.shape == (0,)


def test_track_video_without_a_calibration_exits_with_a_clear_message(tmp_path, monkeypatch):
    """Not a TypeError from np.load(None): the user is told what to pass."""
    import pytest
    from scripts import track_video

    config = tmp_path / "clip.yaml"
    config.write_text("io:\n  input_video: clip.mp4\n")
    monkeypatch.setattr("sys.argv", ["track_video.py", "--config", str(config)])
    with pytest.raises(SystemExit, match="--calibration"):
        track_video.main()
