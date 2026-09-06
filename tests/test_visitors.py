import base64
import json
import math
from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock
from uuid import UUID

import pytest
from conftest import MemoryKeys

from jake.adapters.local_identity_store import LocalIdentityStore
from jake.adapters.person_events import PersonEventGenerator
from jake.adapters.visitor_store import VISITOR_DOMAIN, EncryptedVisitorStore
from jake.config import EventConfig, load_app_config
from jake.domain import BoundingBox, Frame, FrameContext, PersonTrack
from jake.face_identity import FaceIdentityService
from jake.identity import IdentityError, face_cosine, face_normalize
from jake.identity_config import IdentityConfig
from jake.identity_domain import (
    FaceDetection,
    FaceQuality,
    IdentityMatch,
    IdentityState,
    ResidentProfile,
)
from jake.identity_encryption import decrypt
from jake.visitor_config import VisitorConfig
from jake.visitor_domain import (
    VisitorEvent,
    VisitorObservation,
    VisitorProfile,
    VisitorState,
    VisitStatistics,
)
from jake.visitors import VisitorMemory, visitor_match

START = datetime(2026, 1, 1, tzinfo=UTC)
A = face_normalize("test", (1.0, 0.0, 0.0))
B = face_normalize("test", (0.0, 1.0, 0.0))
C = face_normalize("test", (0.0, 0.0, 1.0))
TRACK = PersonTrack("1", BoundingBox(0, 0, 1, 1), 0.99)
CONFIG = VisitorConfig(enabled=True)


class Timeline:
    def __init__(self, store: EncryptedVisitorStore, config: VisitorConfig = CONFIG) -> None:
        self.store = store
        self.counter = 0
        self.memory = VisitorMemory(
            config, store, is_nonresident=lambda _: True, new_id=self.next_id
        )
        self.events = PersonEventGenerator(EventConfig())
        self.sequence = 0
        self.emitted: list[VisitorEvent] = []

    def next_id(self) -> str:
        self.counter += 1
        return str(UUID(int=self.counter))

    def step(
        self,
        seconds: float,
        tracks: tuple[PersonTrack, ...] = (TRACK,),
        observations: dict[str, VisitorObservation] | None = None,
        blocked: set[str] | None = None,
        identities: dict[str, IdentityMatch] | None = None,
    ) -> dict[str, object]:
        context = FrameContext("test", self.sequence, START + timedelta(seconds=seconds))
        self.sequence += 1
        result, events = self.memory.process(
            context,
            tracks,
            self.events.generate(context, tracks),
            observations
            if observations is not None
            else {t.track_id: VisitorObservation(A, 0.99) for t in tracks},
            blocked or set(),
            identities or {},
        )
        self.emitted.extend(events)
        return {k: v.state for k, v in result.items()}

    def confirm(self, start: float = 0, track: PersonTrack = TRACK) -> None:
        for i in range(5):
            self.step(start + i, (track,))


@pytest.fixture
def store(
    tmp_path: Path, memory_keys: MemoryKeys, monkeypatch: pytest.MonkeyPatch
) -> EncryptedVisitorStore:
    monkeypatch.setattr("jake.adapters.local_identity_store.private_permissions", Mock())
    return EncryptedVisitorStore(tmp_path, memory_keys)


def test_disabled_never_reads_or_writes_store() -> None:
    store = Mock()
    memory = VisitorMemory(VisitorConfig(), store, is_nonresident=lambda _: True)
    assert memory.process(
        FrameContext("test", 0, START), (TRACK,), (), {"1": VisitorObservation(A, 0.99)}, set(), {}
    ) == ({}, ())
    assert store.mock_calls == []


def test_single_frame_and_low_quality_do_not_create(store: EncryptedVisitorStore) -> None:
    timeline = Timeline(store)
    assert timeline.step(0)["1"] == VisitorState.VISITOR_CANDIDATE
    for i in range(1, 10):
        timeline.step(i, observations={"1": VisitorObservation(A, 0.80)})
    assert store.profiles() == () and not store._file.path.exists()


def test_first_profile_has_centroid_no_name_and_no_per_frame_visits(
    store: EncryptedVisitorStore,
) -> None:
    t = Timeline(store)
    t.confirm()
    profile = store.profiles()[0]
    assert profile.template == A
    assert (
        profile.visit_count == 1 and profile.display_name is None and not profile.explicitly_labeled
    )
    for i in range(5, 30):
        assert t.step(i)["1"] == VisitorState.FIRST_TIME_VISITOR
    assert store.profiles()[0] == profile
    assert [e.kind for e in t.emitted] == ["VISITOR_FIRST_SEEN"]


def test_reid_same_session_and_final_left_count_once(store: EncryptedVisitorStore) -> None:
    t = Timeline(store)
    t.confirm()
    original = store.profiles()[0].visitor_id
    t.step(5, (replace(TRACK, missed_frames=2, recently_lost=True),), {})
    for i in range(6, 11):
        t.step(i, (replace(TRACK, continuity_epoch=1),))
    assert store.profiles()[0].visitor_id == original and store.profiles()[0].visit_count == 1
    t.step(12, ())
    t.step(13, ())
    profile = store.profiles()[0]
    assert profile.statistics == VisitStatistics(1, 12.0, 0.0)
    assert [e.kind for e in t.emitted] == ["VISITOR_FIRST_SEEN", "VISITOR_LEFT"]
    assert t.memory._visits == {}


def test_separate_visit_recurring_and_welford(store: EncryptedVisitorStore) -> None:
    t = Timeline(store)
    t.confirm()
    t.step(10, ())
    t.confirm(20, replace(TRACK, track_id="2"))
    assert store.profiles()[0].visit_count == 2
    assert visitor_match(store.profiles()[0], 2).state == VisitorState.RECURRING_VISITOR
    t.step(40, ())
    stats = store.profiles()[0].statistics
    assert stats.completed == 2 and stats.mean_seconds == 15 and stats.variance_seconds == 50
    assert [e.kind for e in t.emitted] == [
        "VISITOR_FIRST_SEEN",
        "VISITOR_LEFT",
        "VISITOR_RECOGNIZED",
        "VISITOR_BECAME_RECURRING",
        "VISITOR_LEFT",
    ]


def test_recurring_threshold_configurable(store: EncryptedVisitorStore) -> None:
    t = Timeline(store, replace(CONFIG, recurring_visit_count=3))
    for visit in range(3):
        t.confirm(visit * 20, replace(TRACK, track_id=str(visit + 1)))
        assert visitor_match(store.profiles()[0], 3).state == (
            VisitorState.RECURRING_VISITOR if visit == 2 else VisitorState.FIRST_TIME_VISITOR
        )
        t.step(visit * 20 + 10, ())
    assert sum(e.kind == "VISITOR_BECAME_RECURRING" for e in t.emitted) == 1


@pytest.mark.parametrize("identity", [IdentityState.CANDIDATE, IdentityState.RESIDENT])
def test_resident_state_blocks_entire_visit(
    identity: IdentityState, store: EncryptedVisitorStore
) -> None:
    t = Timeline(store)
    t.step(0, identities={"1": IdentityMatch(identity)})
    for i in range(1, 10):
        assert t.step(i)["1"] == VisitorState.UNKNOWN
    assert store.profiles() == ()


def test_ambiguous_resident_block_survives_unknown_results(store: EncryptedVisitorStore) -> None:
    t = Timeline(store)
    t.step(0, blocked={"1"})
    t.confirm(1)
    assert store.profiles() == ()


def test_face_stage_resident_first_and_quality_gate() -> None:
    detector, encoder, residents = Mock(), Mock(), Mock()
    detector.detect.return_value = (FaceDetection(TRACK.box, 0.99),)
    encoder.encode.return_value = A
    residents.profiles.return_value = (
        ResidentProfile(str(UUID(int=1)), "One", (A,), 5, START),
        ResidentProfile(str(UUID(int=2)), "Two", (A,), 5, START),
    )
    service = FaceIdentityService(
        IdentityConfig(),
        detector,
        encoder,
        residents,
        lambda *_: FaceQuality(True, "ok"),
        collect_visitors=True,
    )
    frame = Frame("test", 0, START, 1, 1, bytes(3))
    matches, _ = service.process(frame, (TRACK,))
    assert matches["1"].state == IdentityState.UNKNOWN  # Ambiguous resident result.
    assert service.visitor_blocked == {"1"} and service.visitor_observations == {}
    encoder.encode.return_value = C
    service.process(replace(frame, sequence=1, captured_at=START + timedelta(seconds=1)), (TRACK,))
    assert service.visitor_observations["1"].embedding == C
    encoder.reset_mock()
    service.quality = lambda *_: FaceQuality(False, "blur")
    service.process(replace(frame, sequence=2, captured_at=START + timedelta(seconds=2)), (TRACK,))
    assert service.visitor_observations == {}
    encoder.encode.assert_not_called()


def test_similar_visitor_ambiguity_remains_unknown(store: EncryptedVisitorStore) -> None:
    for n, embedding in enumerate((A, face_normalize("test", (0.99, 0.1, 0))), 1):
        store.add(VisitorProfile(str(UUID(int=n)), embedding, START, START, START))
    t = Timeline(store)
    for i in range(10):
        assert t.step(i)["1"] == VisitorState.UNKNOWN
    assert [p.visit_count for p in store.profiles()] == [1, 1]


@pytest.mark.parametrize("similar", [True, False])
def test_simultaneous_visitors_no_false_merge(similar: bool, store: EncryptedVisitorStore) -> None:
    t = Timeline(store)
    tracks = (TRACK, replace(TRACK, track_id="2"))
    for i in range(5):
        t.step(
            i,
            tracks,
            {"1": VisitorObservation(A, 0.99), "2": VisitorObservation(A if similar else B, 0.99)},
        )
    assert len(store.profiles()) == (0 if similar else 2)


def test_conflicting_observations_reset_confirmation(store: EncryptedVisitorStore) -> None:
    t = Timeline(store)
    for i in range(20):
        t.step(i, observations={"1": VisitorObservation(A if i % 2 else B, 0.99)})
    assert store.profiles() == ()


def test_label_remove_delete_retention_and_no_resident_mutation(
    store: EncryptedVisitorStore,
) -> None:
    t = Timeline(store)
    t.confirm()
    original = store.profiles()[0]
    store.label(original.visitor_id, "Daniel")
    profile = store.profiles()[0]
    assert visitor_match(profile, 2).state == VisitorState.KNOWN_VISITOR
    assert profile.template == original.template
    store.label(profile.visitor_id, None)
    assert store.profiles()[0] == original
    store.expire(START + timedelta(days=29), 30)
    assert len(store.profiles()) == 1
    store.expire(START + timedelta(days=31), 30)
    assert store.profiles() == ()
    store.add(original)
    store.delete(original.visitor_id)
    assert store.profiles() == ()
    store.add(original)
    store.delete_all()
    assert store.profiles() == ()
    assert not (store._file.root / "residents.json").exists()


def test_encryption_key_domain_and_payload_separation(
    store: EncryptedVisitorStore, memory_keys: MemoryKeys
) -> None:
    resident_store = LocalIdentityStore(store._file.root, memory_keys)
    resident_store.add(ResidentProfile(str(UUID(int=80)), "Resident", (C,), 5, START))
    resident_bytes = resident_store.path.read_bytes()
    Timeline(store).confirm()
    raw = store._file.path.read_bytes()
    document = json.loads(raw)
    assert document["key_id"] != json.loads(resident_bytes)["key_id"]
    assert document["domain"] == VISITOR_DOMAIN
    assert resident_store.path.read_bytes() == resident_bytes
    assert b'"values"' not in raw and b'"visit_count"' not in raw
    payload, _ = decrypt(document, memory_keys, VISITOR_DOMAIN)
    assert EncryptedVisitorStore.parse(payload) == store.profiles()
    with pytest.raises(IdentityError):
        decrypt(document, memory_keys)
    ciphertext = bytearray(base64.b64decode(document["ciphertext"]))
    ciphertext[0] ^= 1
    document["ciphertext"] = base64.b64encode(ciphertext).decode()
    store._file.path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(IdentityError, match="authentication"):
        store.delete_all()
    assert resident_store.path.read_bytes() == resident_bytes


def test_metadata_events_and_fixed_templates(store: EncryptedVisitorStore) -> None:
    t = Timeline(store)
    t.confirm()
    for i in range(5, 15):
        t.step(i, observations={"1": VisitorObservation(face_normalize("test", (1, 0.2, 0)), 0.99)})
    assert store.profiles()[0].template == A
    assert {f.name for f in fields(VisitorEvent)} == {
        "event_id",
        "kind",
        "context",
        "track_id",
        "match",
    }
    assert "values" not in repr(t.emitted) and "embedding" not in repr(t.emitted)


@pytest.mark.parametrize(
    "changes",
    [
        {"enabled": 1},
        {"required_observations": 1},
        {"retention_days": 0},
        {"match_similarity": float("nan")},
        {"ambiguity_margin": 0.9},
        {"observation_window_seconds": 0.1},
        {"recurring_visit_count": 1},
    ],
)
def test_invalid_visitor_config(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        VisitorConfig(**changes)  # type: ignore[arg-type]


def test_config_opt_in_and_cli_management(
    store: EncryptedVisitorStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from jake.visitor_cli import main

    config = tmp_path / "config.toml"
    config.write_text('[pipeline]\ncamera_id="test"\n[visitors]\nenabled=true', encoding="utf-8")
    assert load_app_config(config).visitors.enabled
    Timeline(store).confirm()
    visitor_id = store.profiles()[0].visitor_id
    monkeypatch.setattr(
        "jake.adapters.visitor_store.EncryptedVisitorStore", Mock(return_value=store)
    )
    monkeypatch.setattr(store, "expire", Mock())
    assert main(["--config", str(config), "--label", visitor_id, "--name", "Daniel"]) == 0
    assert main(["--config", str(config), "--list"]) == 0
    assert "Daniel" in capsys.readouterr().out
    assert main(["--config", str(config), "--remove-label", visitor_id]) == 0
    assert store.profiles()[0].display_name is None
    assert main(["--config", str(config), "--delete-all"]) == 0
    assert store.profiles() == ()


def test_welford_known_sequence() -> None:
    stats = VisitStatistics()
    for x in (2, 4, 4, 4, 5, 5, 7, 9):
        stats = stats.add(x)
    assert stats.mean_seconds == 5
    assert stats.variance_seconds == pytest.approx(32 / 7)


def test_deterministic_timeline(
    store: EncryptedVisitorStore, tmp_path: Path, memory_keys: MemoryKeys
) -> None:
    other = EncryptedVisitorStore(tmp_path / "other", memory_keys)
    left, right = Timeline(store), Timeline(other)
    for t in (left, right):
        t.confirm()
        t.step(10, ())
    assert left.emitted == right.emitted
    assert store.profiles() == other.profiles()


@pytest.mark.parametrize("resident_guard", [True, False])
def test_centroid_is_checked_again_before_persistence(
    resident_guard: bool, store: EncryptedVisitorStore
) -> None:
    t = Timeline(store)
    z = 0.44 if resident_guard else 0.59
    if resident_guard:
        t.memory.is_nonresident = lambda e: face_cosine(e, C) < 0.45
    else:
        store.add(VisitorProfile(str(UUID(int=77)), C, START, START, START))
    for i in range(5):
        embedding = face_normalize(
            "test", (math.sqrt(1 - z * z - 0.25**2), 0.25 if i % 2 else -0.25, z)
        )
        t.step(i, observations={"1": VisitorObservation(embedding, 0.99)})
    assert len(store.profiles()) == (0 if resident_guard else 1)
    assert not t.emitted


def test_switching_strong_targets_resets_confirmation(store: EncryptedVisitorStore) -> None:
    other = face_normalize("test", (0.8, 0.6, 0))
    for n, embedding in enumerate((A, other), 60):
        store.add(VisitorProfile(str(UUID(int=n)), embedding, START, START, START))
    t = Timeline(store)
    for i in range(10):
        t.step(i, observations={"1": VisitorObservation(A if i % 2 else other, 0.99)})
    assert [p.visit_count for p in store.profiles()] == [1, 1]


def test_resident_guard_can_reject_direct_evidence(store: EncryptedVisitorStore) -> None:
    t = Timeline(store)
    t.memory.is_nonresident = lambda _: False
    t.confirm()
    assert store.profiles() == ()


def test_occupied_profile_cannot_label_another_track(store: EncryptedVisitorStore) -> None:
    t = Timeline(store)
    t.confirm()
    for i in range(5, 11):
        matches = t.step(i, (TRACK, replace(TRACK, track_id="2")))
        assert matches["2"] == VisitorState.UNKNOWN
    assert store.profiles()[0].visit_count == 1


def test_face_carry_expiry_and_hourly_retention_checkpoint(store: EncryptedVisitorStore) -> None:
    t = Timeline(store)
    t.confirm()
    assert t.step(5, observations={})["1"] == VisitorState.FIRST_TIME_VISITOR
    assert t.step(7, observations={})["1"] == VisitorState.UNKNOWN
    t.step(3700)
    assert store.profiles()[0].last_seen_at == START + timedelta(seconds=3700)


def test_observation_window_and_spacing(store: EncryptedVisitorStore) -> None:
    t = Timeline(store)
    for i in range(10):
        t.step(i * 0.01)
    assert store.profiles() == ()
    for i in range(4):
        t.step(20 + i)
    assert store.profiles() == ()
    t.step(24)
    assert len(store.profiles()) == 1


def test_visitor_store_failure_and_strict_schema(
    store: EncryptedVisitorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    Timeline(store).confirm()
    original = store._file.path.read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(
            "jake.adapters.local_identity_store.os.replace",
            Mock(side_effect=OSError("replace failed")),
        )
        with pytest.raises(OSError):
            store.label(store.profiles()[0].visitor_id, "Daniel")
    assert store._file.path.read_bytes() == original
    assert sorted(p.name for p in store._file.root.iterdir()) == ["visitors.json"]
    with pytest.raises(IdentityError):
        store.add(store.profiles()[0])
    for payload in ({}, {"version": True, "visitors": []}, {"version": 1, "visitors": [{}]}):
        with pytest.raises(IdentityError, match="corrupt"):
            store.parse(payload)


def test_pipeline_visitor_events_stay_separate() -> None:
    from jake.config import PipelineConfig
    from jake.pipeline import PerceptionPipeline

    detector, tracker, events, identity, visitors = (Mock() for _ in range(5))
    detector.detect.return_value = ()
    tracker.update.return_value = (TRACK,)
    events.generate.return_value = ()
    identity.process.return_value = ({}, ())
    identity.visitor_observations = {"1": VisitorObservation(A, 0.99)}
    identity.visitor_blocked = set()
    visitors.process.return_value = ({}, ())
    pipeline = PerceptionPipeline(
        PipelineConfig("test"), detector, tracker, events, identity=identity, visitors=visitors
    )
    assert pipeline.process(Frame("test", 0, START, 1, 1, bytes(3))) == ()
    visitors.process.assert_called_once()
    assert identity.visitor_observations == {}
