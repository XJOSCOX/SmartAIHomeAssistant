import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from jake.adapters.iou_tracker import TrackerError
from jake.adapters.kalman_tracker import AppearanceKalmanPersonTracker
from jake.adapters.openvino_appearance import OpenVINOAppearanceEncoder, load_model
from jake.appearance import (
    AppearanceError,
    cosine_similarity,
    encode_detections,
    normalize,
    update_embedding,
)
from jake.appearance_matching import appearance_cost, assign_appearance_boxes
from jake.config import AppearanceConfig, TrackingConfig, load_app_config
from jake.domain import AppearanceEmbedding, BoundingBox, Frame, FrameContext, PersonDetection

START = datetime(2026, 1, 1, tzinfo=UTC)
A = normalize((1.0, 0.0))
B = normalize((0.0, 1.0))
BOX = BoundingBox(0.1, 0.2, 0.13, 0.7)
CLOSE = BoundingBox(0.2, 0.2, 0.23, 0.7)
CONFIG = TrackingConfig(assignment="hungarian")


def ctx(seq: int, seconds: float) -> FrameContext:
    return FrameContext("test", seq, START + timedelta(seconds=seconds))


def test_math_and_ema() -> None:
    assert cosine_similarity(A, A) == 1
    assert cosine_similarity(A, B) == 0
    result = update_embedding(A, B, 0.8)
    assert result.values == pytest.approx(np.array([0.8, 0.2]) / np.linalg.norm([0.8, 0.2]))
    assert update_embedding(None, B, 0.8) == B
    assert np.linalg.norm(normalize((3.0, 4.0)).values) == pytest.approx(1)
    with pytest.raises(AppearanceError, match="dimensions"):
        cosine_similarity(A, normalize((1.0, 0.0, 0.0)))
    assert "values" not in repr(A)


@pytest.mark.parametrize("values", [(), (0.0, 0.0), (float("nan"), 1.0), (float("inf"), 1.0)])
def test_invalid_vector(values: tuple[float, ...]) -> None:
    with pytest.raises(AppearanceError):
        normalize(values)


@pytest.mark.parametrize("values", [[], (), (True,), (0.0,), (2.0,), (float("nan"),)])
def test_domain_vector_validation(values: object) -> None:
    with pytest.raises(ValueError):
        AppearanceEmbedding(values)  # type: ignore[arg-type]


@pytest.mark.parametrize("strategy", ["greedy", "hungarian"])
def test_gates_one_to_one_and_determinism(strategy: str) -> None:
    config = replace(CONFIG, assignment=strategy)
    assert appearance_cost(BOX, CLOSE, A, A, config) is not None
    assert appearance_cost(BOX, BOX, A, B, config) is None
    assert appearance_cost(BOX, BoundingBox(0.8, 0.2, 0.83, 0.7), A, A, config) is None
    assert appearance_cost(BOX, CLOSE, None, A, config) is None
    expected = assign_appearance_boxes((BOX, BOX), (CLOSE, CLOSE), (A, A), (A, A), config)
    assert len(expected) == len({i for i, _ in expected}) == len({j for _, j in expected}) == 2
    assert expected == assign_appearance_boxes((BOX, BOX), (CLOSE, CLOSE), (A, A), (A, A), config)
    assert assign_appearance_boxes((), (), (), (), config) == ()


def test_lost_reidentification_ema_and_expiry() -> None:
    config = replace(CONFIG, appearance=AppearanceConfig(max_embedding_age_seconds=0.4))
    tracker = AppearanceKalmanPersonTracker(config)
    tracker.update(ctx(0, 0), (PersonDetection(BOX, 0.9, A),))
    tracker.update(ctx(1, 0.1), ())
    shifted = normalize((1.0, 0.1))
    tracks = tracker.update(ctx(2, 0.2), (PersonDetection(CLOSE, 0.9, shifted),))
    assert len(tracks) == 1 and tracks[0].track_id == "1"
    assert tracker._tracks[0].appearance == update_embedding(A, shifted, 0.8)
    assert tracker.diagnostics()[0].appearance_similarity == pytest.approx(
        cosine_similarity(A, shifted)
    )
    tracker.update(ctx(3, 0.601), ())
    assert tracker._tracks[0].appearance is None and tracker._tracks[0].appearance_at is None
    assert tracker.diagnostics()[0].appearance_similarity is None
    assert tracker.update(ctx(4, 2), ()) == ()
    assert tracker.update(ctx(5, 2.1), (PersonDetection(BOX, 0.9, A),))[0].track_id == "2"


def test_incompatible_and_missing_embedding_validation_is_atomic() -> None:
    tracker = AppearanceKalmanPersonTracker(CONFIG)
    tracker.update(ctx(0, 0), (PersonDetection(BOX, 0.9, A),))
    with pytest.raises(TrackerError, match="encoded"):
        tracker.update(ctx(1, 0.1), (PersonDetection(BOX, 0.9),))
    with pytest.raises(TrackerError, match="dimension"):
        tracker.update(ctx(1, 0.1), (PersonDetection(BOX, 0.9, normalize((1.0, 0.0, 0.0))),))
    tracks = tracker.update(ctx(1, 0.1), (PersonDetection(BOX, 0.9, B),))
    assert [t.track_id for t in tracks] == ["1", "2"]
    assert not tracks[0].visible and tracks[1].visible


def test_age_expiry_reinitializes_from_geometry_without_stale_gate() -> None:
    tracker = AppearanceKalmanPersonTracker(
        replace(CONFIG, appearance=AppearanceConfig(max_embedding_age_seconds=0.1))
    )
    tracker.update(ctx(0, 0), (PersonDetection(BOX, 0.9, A),))
    tracks = tracker.update(ctx(1, 0.2), (PersonDetection(BOX, 0.9, B),))
    assert len(tracks) == 1 and tracker._tracks[0].appearance == B


def test_encoder_crop_preprocessing_and_no_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = Mock(return_value=np.ones((1, 256), dtype=np.float32))
    monkeypatch.setattr("jake.adapters.openvino_appearance.load_model", Mock(return_value=runner))
    encoder = OpenVINOAppearanceEncoder(AppearanceConfig())
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[2:, 2:] = [10, 20, 30]
    frame = Frame("test", 0, START, 4, 4, rgb.tobytes())
    before = frame.pixels
    monkeypatch.setattr("builtins.open", Mock(side_effect=AssertionError("no disk writes")))
    result = encoder.encode(frame, BoundingBox(0.5, 0.5, 1, 1))
    tensor = runner.call_args.args[0]
    assert tensor.shape == (1, 3, 256, 128) and tensor.dtype == np.float32
    np.testing.assert_array_equal(tensor[0, :, 0, 0], [30, 20, 10])
    assert len(result.values) == 256 and np.linalg.norm(result.values) == pytest.approx(1)
    tensor[:] = 255
    assert frame.pixels == before
    # Tiny valid corner boxes still include one pixel; no empty resize input.
    encoder.encode(frame, BoundingBox(0.999, 0.999, 1, 1))
    assert encode_detections(frame, (), encoder) == ()
    detection = PersonDetection(BOX, 0.9)
    assert encode_detections(frame, (detection,), encoder)[0].appearance is not None
    assert detection.appearance is None


@pytest.mark.parametrize(
    "output",
    [np.zeros((1, 256)), np.ones((256,)), np.full((1, 256), np.nan), RuntimeError("failed")],
)
def test_encoder_errors(monkeypatch: pytest.MonkeyPatch, output: object) -> None:
    runner = (
        Mock(side_effect=output) if isinstance(output, Exception) else Mock(return_value=output)
    )
    monkeypatch.setattr("jake.adapters.openvino_appearance.load_model", Mock(return_value=runner))
    with pytest.raises(AppearanceError):
        OpenVINOAppearanceEncoder(AppearanceConfig()).encode(
            Frame("test", 0, START, 1, 1, b"abc"), BOX
        )


def test_model_load_boundary_and_privacy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "model.xml"
    with pytest.raises(AppearanceError, match="missing"):
        load_model(path)
    path.write_text("fake", encoding="utf-8")
    path.with_suffix(".bin").write_bytes(b"fake")
    core, model, compiled = Mock(), Mock(), Mock()
    model.inputs = model.outputs = [Mock()]
    model.input.return_value.shape = [1, 3, 256, 128]
    model.output.return_value.shape = [1, 256]
    core.read_model.return_value = model
    core.compile_model.return_value = compiled
    compiled.output.return_value = "embedding"
    compiled.return_value = {"embedding": np.ones((1, 256))}
    monkeypatch.delitem(sys.modules, "openvino", raising=False)
    monkeypatch.setitem(sys.modules, "openvino_telemetry", Mock())
    importer = Mock(return_value=Mock(Core=Mock(return_value=core)))
    monkeypatch.setattr("jake.adapters.openvino_appearance.import_module", importer)
    run = load_model(path)
    assert sys.modules["openvino_telemetry"] is None
    core.compile_model.assert_called_once_with(model, "CPU", {"CACHE_DIR": ""})
    assert run(np.ones((1, 3, 256, 128), dtype=np.float32)).shape == (1, 256)
    model.input.return_value.shape = [1, 3, 128, 256]
    with pytest.raises(AppearanceError, match="Expected"):
        load_model(path)
    importer.side_effect = ImportError("unavailable")
    with pytest.raises(AppearanceError, match="Cannot load"):
        load_model(path)
    monkeypatch.setitem(sys.modules, "openvino", Mock())
    monkeypatch.setitem(sys.modules, "openvino_telemetry", Mock())
    with pytest.raises(AppearanceError, match="fresh"):
        load_model(path)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(enabled=1),
        dict(model="http://model.xml"),
        dict(weight=0),
        dict(min_similarity=-0.1),
        dict(ema_alpha=1),
        dict(max_embedding_age_seconds=0),
        dict(weight=float("nan")),
        dict(weight=True),
        dict(weight=".35"),
    ],
)
def test_invalid_config(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        AppearanceConfig(**kwargs)  # type: ignore[arg-type]


def test_config_loading(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[pipeline]\ncamera_id="test"\n[tracking.appearance]\nenabled=true\nweight=0.5',
        encoding="utf-8",
    )
    assert load_app_config(path).tracking.appearance == AppearanceConfig(enabled=True, weight=0.5)
    path.write_text(
        '[pipeline]\ncamera_id="test"\n[tracking.appearance]\nunknown=true', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="appearance"):
        load_app_config(path)


@pytest.mark.parametrize(
    "scenario", ["occlusion", "edge", "low-iou", "crossing", "appearance-noise", "similar-outfits"]
)
def test_benchmark_regression(scenario: str) -> None:
    from jake.appearance_benchmark import compare

    geometry, appearance = compare(scenario, False), compare(scenario, True)
    assert appearance == compare(scenario, True)
    if scenario == "similar-outfits":
        assert appearance.id_switches == geometry.id_switches == 2
        assert appearance.false_reassociations == 18
    else:
        assert (
            appearance.id_switches
            == appearance.fragmentation
            == appearance.false_reassociations
            == 0
        )
        assert geometry.id_switches > 0
    assert appearance.assignment_ms >= 0


def test_appearance_pipeline_sees_filtered_crops_only() -> None:
    from jake.adapters.person_events import PersonEventGenerator
    from jake.config import EventConfig, PipelineConfig
    from jake.pipeline import PerceptionPipeline

    detector, encoder = Mock(), Mock()
    detector.detect.return_value = (PersonDetection(BOX, 0.9), PersonDetection(CLOSE, 0.1))
    encoder.encode.return_value = A
    pipeline = PerceptionPipeline(
        PipelineConfig("test"),
        detector,
        AppearanceKalmanPersonTracker(replace(CONFIG, confirmation_hits=1)),
        PersonEventGenerator(EventConfig()),
        encoder=encoder,
    )
    frame = Frame("test", 0, START, 1, 1, b"abc")
    events = pipeline.process(frame)
    encoder.encode.assert_called_once_with(frame, BOX)
    assert len(events) == 1 and "appearance" not in repr(events)
