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
    ResidentEvidence,
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
        resident_evidence: dict[str, ResidentEvidence] | None = None,
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
            resident_evidence=resident_evidence,
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
        profile.session_count == 1
        and profile.display_name is None
        and not profile.explicitly_labeled
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
    assert store.profiles()[0].visitor_id == original and store.profiles()[0].session_count == 1
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
    assert store.profiles()[0].session_count == 2
    assert visitor_match(store.profiles()[0], CONFIG).state == VisitorState.FIRST_TIME_VISITOR
    t.step(40, ())
    stats = store.profiles()[0].statistics
    assert stats.completed == 2 and stats.mean_seconds == 15 and stats.variance_seconds == 50
    assert [e.kind for e in t.emitted] == [
        "VISITOR_FIRST_SEEN",
        "VISITOR_LEFT",
        "VISITOR_RECOGNIZED",
        "VISITOR_LEFT",
    ]


def test_recurring_threshold_configurable(store: EncryptedVisitorStore) -> None:
    t = Timeline(store, replace(CONFIG, recurring_distinct_days=3))
    for visit in range(3):
        t.confirm(visit * 86400, replace(TRACK, track_id=str(visit + 1)))
        assert visitor_match(store.profiles()[0], t.memory.config).state == (
            VisitorState.RECURRING_VISITOR if visit == 2 else VisitorState.FIRST_TIME_VISITOR
        )
        t.step(visit * 86400 + 10, ())
    assert sum(e.kind == "VISITOR_BECAME_RECURRING" for e in t.emitted) == 1


@pytest.mark.parametrize("identity", [IdentityState.RESIDENT])
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
        store.add(
            VisitorProfile(
                str(UUID(int=n)), embedding, START, START, START, last_visit_local_date=START.date()
            )
        )
    t = Timeline(store)
    for i in range(10):
        assert t.step(i)["1"] == VisitorState.UNKNOWN
    assert [p.session_count for p in store.profiles()] == [1, 1]


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
    assert visitor_match(profile, CONFIG).state == VisitorState.KNOWN_VISITOR
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
    assert b'"values"' not in raw and b'"session_count"' not in raw
    payload, _ = decrypt(document, memory_keys, VISITOR_DOMAIN)
    assert store.parse(payload) == store.profiles()
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
        {"recurring_distinct_days": 1},
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
        store.add(
            VisitorProfile(
                str(UUID(int=77)), C, START, START, START, last_visit_local_date=START.date()
            )
        )
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
        store.add(
            VisitorProfile(
                str(UUID(int=n)), embedding, START, START, START, last_visit_local_date=START.date()
            )
        )
    t = Timeline(store)
    for i in range(10):
        t.step(i, observations={"1": VisitorObservation(A if i % 2 else other, 0.99)})
    assert [p.session_count for p in store.profiles()] == [1, 1]


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
    assert store.profiles()[0].session_count == 1


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


@pytest.mark.parametrize("transient", ["candidate", "blocked"])
def test_transient_resident_evidence_recovers(transient: str, store: EncryptedVisitorStore) -> None:
    t = Timeline(store)
    t.step(
        0,
        blocked={"1"} if transient == "blocked" else set(),
        identities={"1": IdentityMatch(IdentityState.CANDIDATE)}
        if transient == "candidate"
        else {},
    )
    assert "paused possible resident" in t.memory.diagnostics["1"]
    for i in range(1, 4):
        t.step(i)
        assert f"recovering nonresident {i}/3" == t.memory.diagnostics["1"]
        assert store.profiles() == ()
    for i in range(4, 8):
        t.step(i)
        assert t.memory.diagnostics["1"] == f"candidate {i - 3}/5"
    t.step(8)
    assert t.memory.diagnostics["1"] == "confirmed FIRST_TIME_VISITOR"
    assert len(store.profiles()) == 1


def test_repeated_strong_fresh_resident_evidence_blocks(store: EncryptedVisitorStore) -> None:
    t = Timeline(store)
    evidence = ResidentEvidence(str(UUID(int=99)), 0.80, True)
    for i in range(3):
        t.step(i, observations={}, blocked={"1"}, resident_evidence={"1": evidence})
    assert t.memory._visits["1"].resident_candidate_count == 3
    for i in range(3, 15):
        assert t.step(i)["1"] == VisitorState.UNKNOWN
        assert t.memory.diagnostics["1"] == "blocked repeated strong resident evidence"
    assert store.profiles() == ()


def test_carried_candidate_is_not_fresh_evidence(store: EncryptedVisitorStore) -> None:
    t = Timeline(store)
    evidence = ResidentEvidence(str(UUID(int=99)), 0.80, True)
    t.step(0, blocked={"1"}, resident_evidence={"1": evidence})
    for i in range(1, 20):
        t.step(i, observations={}, identities={"1": IdentityMatch(IdentityState.CANDIDATE)})
    assert t.memory._visits["1"].resident_candidate_count == 1
    for i in range(20, 28):
        t.step(i)
    assert len(store.profiles()) == 1


def test_strong_evidence_requires_same_resident_and_spacing(store: EncryptedVisitorStore) -> None:
    t = Timeline(store)
    for i in range(8):
        t.step(
            i * 0.01,
            blocked={"1"},
            resident_evidence={"1": ResidentEvidence(str(UUID(int=99)), 0.8, True)},
        )
    assert t.memory._visits["1"].resident_candidate_count == 1
    t.step(
        1, blocked={"1"}, resident_evidence={"1": ResidentEvidence(str(UUID(int=100)), 0.8, True)}
    )
    assert t.memory._visits["1"].resident_candidate_count == 1
    t.step(
        10, blocked={"1"}, resident_evidence={"1": ResidentEvidence(str(UUID(int=100)), 0.8, True)}
    )
    assert t.memory._visits["1"].resident_candidate_count == 1


def test_low_confidence_diagnostics_and_later_progress(store: EncryptedVisitorStore) -> None:
    t = Timeline(store)
    for i in range(5):
        t.step(i, observations={"1": VisitorObservation(A, 0.87)})
        assert t.memory.diagnostics["1"] == "rejected detector confidence 0.87 < 0.90"
    assert store.profiles() == ()
    for i in range(5, 10):
        t.step(i, observations={"1": VisitorObservation(A, 0.92)})
    assert len(store.profiles()) == 1


def test_recovery_does_not_increment_existing_visit(store: EncryptedVisitorStore) -> None:
    t = Timeline(store)
    t.confirm()
    t.step(5, blocked={"1"})
    for i in range(6, 14):
        t.step(i)
    assert store.profiles()[0].session_count == 1


def test_visitor_confidence_config_alias(tmp_path: Path) -> None:
    path = tmp_path / "settings.toml"
    prefix = '[pipeline]\ncamera_id="test"\n[visitors]\n'
    path.write_text(prefix + "min_face_quality=0.95", encoding="utf-8")
    assert load_app_config(path).visitors.min_detector_confidence == 0.95
    path.write_text(prefix + "min_detector_confidence=0.90", encoding="utf-8")
    assert load_app_config(path).visitors.min_detector_confidence == 0.90
    path.write_text(
        prefix + "min_face_quality=0.95\nmin_detector_confidence=0.90", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="legacy alias"):
        load_app_config(path)


def test_face_rejection_diagnostics_have_no_vectors() -> None:
    detector, encoder, residents = Mock(), Mock(), Mock()
    detector.detect.return_value = (FaceDetection(TRACK.box, 0.87),)
    residents.profiles.return_value = ()
    service = FaceIdentityService(
        IdentityConfig(),
        detector,
        encoder,
        residents,
        lambda *_: FaceQuality(False, "face too small"),
        collect_visitors=True,
    )
    frame = Frame("test", 0, START, 1, 1, bytes(3))
    service.process(frame, (TRACK,))
    assert service.visitor_diagnostics["1"] == "rejected detector confidence 0.87 < 0.90"
    detector.detect.return_value = (FaceDetection(TRACK.box, 0.99),)
    service.process(replace(frame, sequence=1, captured_at=START + timedelta(seconds=1)), (TRACK,))
    assert service.visitor_diagnostics["1"] == "rejected face too small"
    encoder.encode.assert_not_called()
    assert "values" not in repr(service.visitor_diagnostics)


def test_same_day_five_sessions_then_five_days(store: EncryptedVisitorStore) -> None:
    t = Timeline(store)
    for session in range(5):
        t.confirm(session * 600, replace(TRACK, track_id=str(session + 1)))
        t.step(session * 600 + 10, ())
        p = store.profiles()[0]
        assert (p.session_count, p.distinct_visit_days) == (session + 1, 1)
        assert visitor_match(p, CONFIG).state == VisitorState.FIRST_TIME_VISITOR
    for day in range(1, 5):
        t.confirm(day * 86400, replace(TRACK, track_id=str(day + 10)))
        t.step(day * 86400 + 10, ())
        p = store.profiles()[0]
        assert (p.session_count, p.distinct_visit_days) == (5 + day, 1 + day)
        assert visitor_match(p, CONFIG).state == (
            VisitorState.FREQUENT_VISITOR if day == 4 else VisitorState.RECURRING_VISITOR
        )
        assert p.display_name is None
    t.confirm(4 * 86400 + 600, replace(TRACK, track_id="99"))
    t.step(4 * 86400 + 610, ())
    assert sum(e.kind == "VISITOR_BECAME_RECURRING" for e in t.emitted) == 1
    assert sum(e.kind == "VISITOR_BECAME_FREQUENT" for e in t.emitted) == 1
    assert store.profiles()[0].statistics.completed == 10
    assert store.profiles()[0].statistics.mean_seconds == 10
    assert store.profiles()[0].statistics.variance_seconds == 0
    assert "values" not in repr(t.emitted) and "template" not in repr(t.emitted)


@pytest.mark.parametrize(
    ("first", "second", "days"),
    [
        ("2026-01-01T23:55:00+00:00", "2026-01-02T00:10:00+00:00", 1),
        ("2026-01-02T05:55:00+00:00", "2026-01-02T06:10:00+00:00", 2),
        ("2026-03-08T07:55:00+00:00", "2026-03-08T08:10:00+00:00", 1),
        ("2026-11-01T06:55:00+00:00", "2026-11-01T07:10:00+00:00", 1),
    ],
)
def test_household_calendar_and_dst(
    store: EncryptedVisitorStore, first: str, second: str, days: int
) -> None:
    from zoneinfo import ZoneInfo

    store.timezone = ZoneInfo("America/Chicago")
    t = Timeline(store)
    t.memory.timezone = store.timezone
    a = (datetime.fromisoformat(first) - START).total_seconds()
    b = (datetime.fromisoformat(second) - START).total_seconds()
    t.confirm(a)
    t.step(a + 10, ())
    t.confirm(b, replace(TRACK, track_id="2"))
    p = store.profiles()[0]
    assert p.session_count == 2 and p.distinct_visit_days == days
    assert (
        p.last_visit_local_date == datetime.fromisoformat(second).astimezone(store.timezone).date()
    )


def test_confirmation_day_and_overnight_presence(store: EncryptedVisitorStore) -> None:
    t = Timeline(store)
    t.confirm(86398)  # Confirmation crosses midnight; entry date is not counted.
    assert store.profiles()[0].last_visit_local_date == (START + timedelta(days=1)).date()
    t.step(2 * 86400)
    assert store.profiles()[0].distinct_visit_days == 1
    assert store.profiles()[0].session_count == 1


def test_label_precedence_counts_continue(store: EncryptedVisitorStore) -> None:
    t = Timeline(store)
    t.confirm()
    visitor_id = store.profiles()[0].visitor_id
    store.label(visitor_id, "Daniel")
    t.step(10, ())
    for day in range(1, 5):
        t.confirm(day * 86400, replace(TRACK, track_id=str(day + 10)))
        t.step(day * 86400 + 10, ())
        assert visitor_match(store.profiles()[0], CONFIG).state == VisitorState.KNOWN_VISITOR
    p = store.profiles()[0]
    assert p.session_count == p.distinct_visit_days == 5
    store.label(visitor_id, None)
    assert visitor_match(store.profiles()[0], CONFIG).state == VisitorState.FREQUENT_VISITOR


@pytest.mark.parametrize(
    "overrides",
    [
        {"recurring_distinct_days": 1},
        {"frequent_distinct_days": 2},
        {"recurring_distinct_days": 5, "frequent_distinct_days": 5},
        {"frequent_distinct_days": "5"},
        {"recurring_distinct_days": True},
        {"recurrence_policy": "minimum_gap"},
        {"frequent_distinct_days": float("nan")},
    ],
)
def test_frequency_config_rejects_invalid(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        VisitorConfig(**overrides)  # type: ignore[arg-type]


@pytest.mark.parametrize("timezone", ["Invalid/Home", "", 3])
def test_home_timezone_invalid(timezone: object) -> None:
    from jake.home_config import HomeConfig

    with pytest.raises(ValueError):
        HomeConfig(timezone)  # type: ignore[arg-type]


def legacy_document(store: EncryptedVisitorStore, keys: MemoryKeys) -> dict[str, object]:
    """Build the actual old schema, never a plaintext on-disk fixture."""
    from jake.identity_encryption import encrypt

    profile = VisitorProfile(
        str(UUID(int=123)),
        A,
        START,
        START,
        START,
        14,
        VisitStatistics(2, 15, 50),
        "Daniel",
        True,
        last_visit_local_date=START.date(),
    )
    payload = store.payload((profile,))
    assert isinstance(payload, dict)
    payload.pop("timezone")
    payload["version"] = 1
    for row in payload["visitors"]:
        row["visit_count"] = row.pop("session_count")
        row.pop("distinct_visit_days")
        row.pop("last_visit_local_date")
    key_id, _ = keys.create()
    document = encrypt(payload, key_id, keys, VISITOR_DOMAIN)
    store._file.path.write_bytes(document)
    return payload


def test_explicit_legacy_migration_fidelity_and_idempotence(
    store: EncryptedVisitorStore, memory_keys: MemoryKeys
) -> None:
    old = legacy_document(store, memory_keys)
    before = store._file.path.read_bytes()
    with pytest.raises(IdentityError, match="explicit migration"):
        store.profiles()
    assert store._file.path.read_bytes() == before
    key_ids = set(memory_keys.keys)
    assert "migrated" in store.migrate()
    p = store.profiles()[0]
    assert p.session_count == 14 and p.distinct_visit_days == 1
    assert p.visitor_id == str(UUID(int=123)) and p.template == A
    assert p.created_at == p.last_seen_at == p.last_visit_at == START
    assert p.display_name == "Daniel" and p.explicitly_labeled
    assert p.statistics == VisitStatistics(2, 15, 50)
    assert p.last_visit_local_date == START.date()
    migrated, _ = decrypt(json.loads(store._file.path.read_bytes()), memory_keys, VISITOR_DOMAIN)
    assert isinstance(migrated, dict) and isinstance(old["visitors"], list)
    assert migrated["visitors"][0]["template"] == old["visitors"][0]["template"]
    after = store._file.path.read_bytes()
    assert "already" in store.migrate()
    assert store._file.path.read_bytes() == after
    assert set(memory_keys.keys) == key_ids
    assert b"Daniel" not in after and b"values" not in after
    store.label(p.visitor_id, None)
    assert visitor_match(store.profiles()[0], CONFIG).state == VisitorState.FIRST_TIME_VISITOR


def test_migration_atomic_failure_preserves_original(
    store: EncryptedVisitorStore, memory_keys: MemoryKeys, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy_document(store, memory_keys)
    before = store._file.path.read_bytes()
    keys = dict(memory_keys.keys)
    monkeypatch.setattr("jake.adapters.local_identity_store.os.replace", Mock(side_effect=OSError))
    with pytest.raises(OSError):
        store.migrate()
    assert store._file.path.read_bytes() == before
    assert memory_keys.keys == keys
    assert list(store._file.root.glob(".residents-*")) == []


def test_migration_tampered_ciphertext_fails_closed(
    store: EncryptedVisitorStore, memory_keys: MemoryKeys
) -> None:
    legacy_document(store, memory_keys)
    document = json.loads(store._file.path.read_text())
    document["ciphertext"] = base64.b64encode(b"invalid ciphertext").decode()
    store._file.path.write_text(json.dumps(document))
    before = store._file.path.read_bytes()
    with pytest.raises(IdentityError):
        store.migrate()
    assert store._file.path.read_bytes() == before


def test_timezone_change_rejected_without_write(store: EncryptedVisitorStore) -> None:
    from zoneinfo import ZoneInfo

    Timeline(store).confirm()
    before = store._file.path.read_bytes()
    store.timezone = ZoneInfo("America/Chicago")
    with pytest.raises(IdentityError, match="timezone"):
        store.profiles()
    assert store._file.path.read_bytes() == before


def test_visitor_debug_suppresses_idle_alternation_and_preserves_progress() -> None:
    from jake.visitor_diagnostics import VisitorDiagnostics

    log = VisitorDiagnostics()
    assert log.changes({"1": "waiting for face observation"})
    for reason in (
        "candidate 1/5",
        "candidate 2/5",
        "rejected face too small",
        "confirmed FIRST_TIME_VISITOR",
        "confirmed FREQUENT_VISITOR",
    ):
        assert log.changes({"1": reason}) == (f"VISITOR track=1 {reason}",)
        for _ in range(10):
            assert log.changes({"1": "waiting for face observation"}) == ()
            assert log.changes({"1": reason}) == ()
    assert log.changes({}) == ()
    assert log._previous == {}
    assert log.changes({"1": "candidate 1/5"})


def test_migration_uses_household_date_and_skips_retention(
    store: EncryptedVisitorStore,
    memory_keys: MemoryKeys,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from zoneinfo import ZoneInfo

    from jake.visitor_cli import main

    legacy_document(store, memory_keys)
    store.timezone = ZoneInfo("America/Chicago")
    monkeypatch.setattr(
        "jake.adapters.visitor_store.EncryptedVisitorStore", Mock(return_value=store)
    )
    expire = Mock(side_effect=AssertionError("migration must never prune real profiles"))
    monkeypatch.setattr(store, "expire", expire)
    config = tmp_path / "config.toml"
    config.write_text('[pipeline]\ncamera_id="test"\n[home]\ntimezone="America/Chicago"')
    assert main(["--config", str(config), "--migrate-store"]) == 0
    assert "migrated" in capsys.readouterr().out
    assert store.profiles()[0].last_visit_local_date == (START - timedelta(days=1)).date()
    expire.assert_not_called()


def test_migration_readback_failure_preserves_original(
    store: EncryptedVisitorStore, memory_keys: MemoryKeys, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy_document(store, memory_keys)
    before = store._file.path.read_bytes()
    monkeypatch.setattr(
        "jake.adapters.local_identity_store.decrypt",
        Mock(side_effect=IdentityError("verification failure")),
    )
    with pytest.raises(IdentityError, match="verification"):
        store.migrate()
    assert store._file.path.read_bytes() == before
    assert list(store._file.root.glob(".residents-*")) == []
    assert len(memory_keys.keys) == 1


@pytest.mark.parametrize("version", [1, 2, 3, True])
def test_authenticated_malformed_schema_never_migrates(
    store: EncryptedVisitorStore, memory_keys: MemoryKeys, version: object
) -> None:
    from jake.identity_encryption import encrypt

    key_id, _ = memory_keys.create()
    raw = encrypt({"version": version, "visitors": [{}]}, key_id, memory_keys, VISITOR_DOMAIN)
    store._file.path.write_bytes(raw)
    with pytest.raises(IdentityError, match="corrupt"):
        store.migrate()
    assert store._file.path.read_bytes() == raw


def test_frequency_management_uses_config_thresholds(
    store: EncryptedVisitorStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from jake.visitor_cli import main

    Timeline(store).confirm()
    p = store.profiles()[0]
    store.update(p.visitor_id, lambda p: replace(p, session_count=12, distinct_visit_days=4))
    monkeypatch.setattr(
        "jake.adapters.visitor_store.EncryptedVisitorStore", Mock(return_value=store)
    )
    monkeypatch.setattr(store, "expire", Mock())
    config = tmp_path / "config.toml"
    config.write_text('[pipeline]\ncamera_id="test"\n[visitors]\nfrequent_distinct_days=4')
    assert main(["--config", str(config), "--list"]) == 0
    out = capsys.readouterr().out
    assert "FREQUENT_VISITOR" in out and "Sessions=12" in out and "Visit Days=4" in out
    assert "completed sessions=" in out and "values" not in out and "template" not in out


@pytest.mark.parametrize(
    "table",
    ["[home]\nunknown=3", '[home]\ntimezone="Bad/Zone"', "[visitors]\nrecurring_visit_count=2"],
)
def test_calendar_configuration_errors(tmp_path: Path, table: str) -> None:
    config = tmp_path / "bad.toml"
    config.write_text('[pipeline]\ncamera_id="test"\n' + table)
    with pytest.raises(ValueError):
        load_app_config(config)


def test_restart_clock_rollback_does_not_recount_days(store: EncryptedVisitorStore) -> None:
    t = Timeline(store)
    t.confirm(86400)
    t.step(86410, ())
    t = Timeline(store)  # A new process can have an older but aware frame timeline.
    t.confirm()
    p = store.profiles()[0]
    assert p.session_count == 2 and p.distinct_visit_days == 1
    assert p.last_visit_local_date == (START + timedelta(days=1)).date()


@pytest.mark.parametrize(
    "changes",
    [
        {"session_count": "2"},
        {"session_count": False},
        {"session_count": 0},
        {"distinct_visit_days": 0},
        {"distinct_visit_days": 2},
        {"distinct_visit_days": True},
        {"last_visit_local_date": None},
        {"last_visit_local_date": "2026-01-01"},
        {"last_seen_at": START.replace(tzinfo=None)},
    ],
)
def test_invalid_frequency_profile_metadata(changes: dict[str, object]) -> None:
    profile = VisitorProfile(
        str(UUID(int=1)), A, START, START, START, last_visit_local_date=START.date()
    )
    with pytest.raises(ValueError):
        replace(profile, **changes)  # type: ignore[arg-type]


def test_empty_store_migration_does_not_create_key(
    store: EncryptedVisitorStore, memory_keys: MemoryKeys
) -> None:
    with pytest.raises(IdentityError, match="does not exist"):
        store.migrate()
    assert memory_keys.keys == {} and not store._file.path.exists()
