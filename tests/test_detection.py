"""The detector interface and the YOLO wrapper's translation logic, over a fake model."""

from __future__ import annotations

import numpy as np
import pytest

from football_tracker.detection.interface import Detections, Detector, ObjectClass
from football_tracker.detection.yolo import YOLODetector, build_id_map, translate


class TestObjectClass:
    def test_canonical_ids_are_stable(self):
        # Downstream code relies on these exact integers — pin them.
        assert (ObjectClass.BALL, ObjectClass.GOALKEEPER) == (0, 1)
        assert (ObjectClass.PLAYER, ObjectClass.REFEREE) == (2, 3)


class TestDetections:
    def test_empty(self):
        det = Detections.empty()
        assert len(det) == 0
        assert det.xyxy.shape == (0, 4)
        assert det.foot_points.shape == (0, 2)

    def test_construction_normalizes_dtypes_and_shape(self):
        det = Detections(xyxy=[[0, 0, 10, 20]], confidence=[0.9], class_id=[2])
        assert det.xyxy.dtype == np.float32
        assert det.class_id.dtype == np.int64
        assert len(det) == 1

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="Mismatched lengths"):
            Detections(xyxy=[[0, 0, 1, 1]], confidence=[0.5, 0.6], class_id=[2])

    def test_foot_point_is_bottom_center(self):
        det = Detections(xyxy=[[10, 20, 30, 80]], confidence=[0.9], class_id=[2])
        # x = (10+30)/2 = 20 ; y = bottom = 80
        assert det.foot_points.tolist() == [[20.0, 80.0]]

    def test_filter_by_class(self):
        det = Detections(
            xyxy=[[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]],
            confidence=[0.9, 0.8, 0.7],
            class_id=[ObjectClass.BALL, ObjectClass.PLAYER, ObjectClass.REFEREE],
        )
        players = det.filter_by_class(ObjectClass.PLAYER, ObjectClass.REFEREE)
        assert len(players) == 2
        assert set(players.class_id.tolist()) == {ObjectClass.PLAYER, ObjectClass.REFEREE}

    def test_with_min_confidence(self):
        det = Detections(xyxy=[[0, 0, 1, 1], [0, 0, 1, 1]], confidence=[0.9, 0.2], class_id=[2, 2])
        assert len(det.with_min_confidence(0.5)) == 1


class TestBuildIdMap:
    def test_maps_names_case_insensitively(self):
        # A model whose class order differs from ours (ball is id 3 here).
        names = {0: "player", 1: "Goalkeeper", 2: "Referee", 3: "BALL"}
        id_map = build_id_map(names)
        assert id_map == {
            0: ObjectClass.PLAYER,
            1: ObjectClass.GOALKEEPER,
            2: ObjectClass.REFEREE,
            3: ObjectClass.BALL,
        }

    def test_unknown_names_are_dropped(self):
        id_map = build_id_map({0: "player", 1: "crowd", 2: "ball"})
        assert set(id_map) == {0, 2}


class TestTranslate:
    def test_remaps_native_ids_and_drops_unknown(self):
        # Native model: 0=player, 1=ball, 2=something-we-don't-model.
        id_map = {0: ObjectClass.PLAYER, 1: ObjectClass.BALL}
        det = translate(
            xyxy=np.array([[0, 0, 5, 5], [1, 1, 2, 2], [3, 3, 4, 4]], dtype=float),
            conf=np.array([0.9, 0.8, 0.7]),
            cls=np.array([0, 1, 2]),
            id_map=id_map,
        )
        assert len(det) == 2  # the unknown class-2 detection is dropped
        assert det.class_id.tolist() == [ObjectClass.PLAYER, ObjectClass.BALL]

    def test_ball_confidence_override(self):
        # A stricter threshold for the ball rejects the low-scoring ball detection
        # without touching the player detection.
        id_map = {0: ObjectClass.PLAYER, 1: ObjectClass.BALL}
        det = translate(
            xyxy=np.array([[0, 0, 5, 5], [1, 1, 2, 2]], dtype=float),
            conf=np.array([0.30, 0.30]),
            cls=np.array([0, 1]),
            id_map=id_map,
            ball_conf=0.5,
        )
        assert det.class_id.tolist() == [ObjectClass.PLAYER]

    def test_empty_input(self):
        det = translate(
            xyxy=np.zeros((0, 4)),
            conf=np.zeros((0,)),
            cls=np.zeros((0,), dtype=int),
            id_map={0: ObjectClass.PLAYER},
        )
        assert len(det) == 0


class _FakeBoxes:
    """Stands in for an Ultralytics Results.boxes object."""

    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.asarray(xyxy, dtype=float).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=float).reshape(-1)
        self.cls = np.asarray(cls, dtype=float).reshape(-1)

    def __len__(self):  # real Ultralytics Boxes supports len()
        return len(self.cls)


class _FakeResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class _FakeModel:
    """A stand-in Ultralytics YOLO model that returns canned results."""

    def __init__(self, names, result):
        self.names = names
        self._result = result
        self.predict_kwargs: dict = {}

    def predict(self, frame, **kwargs):
        self.predict_kwargs = kwargs
        return [self._result]


class TestYOLODetector:
    def _detector(self):
        names = {0: "player", 1: "ball", 2: "goalkeeper", 3: "referee"}
        boxes = _FakeBoxes(xyxy=[[0, 0, 10, 40], [5, 5, 7, 7]], conf=[0.9, 0.8], cls=[0, 1])
        model = _FakeModel(names, _FakeResult(boxes, names))
        det = YOLODetector(model, imgsz=1280, conf=0.25, device="cpu")
        return det, model

    def test_satisfies_detector_protocol(self):
        det, _ = self._detector()
        assert isinstance(det, Detector)

    def test_detect_translates_and_passes_config_through(self):
        det, model = self._detector()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = det.detect(frame)
        assert len(result) == 2
        assert result.class_id.tolist() == [ObjectClass.PLAYER, ObjectClass.BALL]
        assert model.predict_kwargs["imgsz"] == 1280
        assert model.predict_kwargs["conf"] == 0.25

    def test_reconfigured_shares_model_and_overrides_inference_params(self):
        det, model = self._detector()
        roi = det.reconfigured(imgsz=640, conf=0.10, ball_conf=None)
        roi.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        # Same underlying model instance — no second weight load.
        assert model.predict_kwargs["imgsz"] == 640
        assert model.predict_kwargs["conf"] == 0.10
        # The original wrapper is untouched.
        det.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        assert model.predict_kwargs["imgsz"] == 1280
        assert model.predict_kwargs["conf"] == 0.25

    def test_detect_on_empty_result(self):
        names = {0: "player"}
        model = _FakeModel(names, _FakeResult(_FakeBoxes([], [], []), names))
        det = YOLODetector(model, imgsz=640, conf=0.25, device="cpu")
        out = det.detect(np.zeros((10, 10, 3), dtype=np.uint8))
        assert len(out) == 0


class TestAgnosticNMS:
    """Class-aware NMS keeps a `player` and a `referee` box on one person."""

    def _model(self):
        names = {0: "player"}
        return _FakeModel(names, _FakeResult(_FakeBoxes([], [], []), names))

    def test_the_flag_reaches_the_model(self):
        model = self._model()
        YOLODetector(model, agnostic_nms=True).detect(np.zeros((8, 8, 3), np.uint8))
        assert model.predict_kwargs["agnostic_nms"] is True

    def test_it_defaults_on(self):
        from football_tracker.config import DetectionConfig

        assert DetectionConfig().agnostic_nms is True

    def test_reconfigured_carries_it_across_to_the_ball_pass(self):
        wrapper = YOLODetector(self._model(), agnostic_nms=True)
        assert wrapper.reconfigured(conf=0.05)._agnostic_nms is True
