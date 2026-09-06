"""Deliberate, consenting resident enrollment; no automatic visitor enrollment."""

import argparse
import sys
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

from jake.config import load_app_config
from jake.domain import BoundingBox
from jake.identity import Enrollment, IdentityError


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Local resident enrollment: stores biometric templates, never images"
    )
    parser.add_argument("--config", type=Path, default=Path("config/local.toml"))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--name")
    group.add_argument("--delete", metavar="RESIDENT_UUID")
    group.add_argument("--list", action="store_true")
    group.add_argument("--migrate-store", action="store_true", help="Explicitly encrypt a v1 store")
    parser.add_argument(
        "--consent",
        action="store_true",
        help="Confirm this resident knowingly consents to local biometric enrollment",
    )
    args = parser.parse_args(argv)
    if args.name and not args.consent:
        parser.error("--consent is required for deliberate biometric enrollment")
    try:
        from jake.adapters.local_identity_store import LocalIdentityStore

        config = load_app_config(args.config)
        store = LocalIdentityStore(Path(config.identity.store_path))
        if args.migrate_store:
            store.migrate()
            print("Identity store migrated to encrypted v2. Restart identity sessions.")
            return 0
        profiles = store.profiles()
        if args.list:
            for profile in profiles:
                print(f"{profile.resident_id} {profile.display_name}")
            return 0
        if args.delete:
            store.delete(args.delete)
            print(
                "Resident templates deleted. Restart live identity sessions to clear cached state."
            )
            return 0
        name = args.name.strip()
        if not name or len(name) > 80 or not name.isprintable():
            raise IdentityError("resident name must be printable and 1-80 characters")
        if any(p.display_name.casefold() == name.casefold() for p in profiles):
            raise IdentityError("resident name already enrolled; delete before re-enrollment")
        import cv2

        from jake.adapters.opencv_camera import OpenCVCamera
        from jake.adapters.opencv_faces import SFaceEncoder, YuNetFaceDetector, crop, face_quality

        detector, encoder = YuNetFaceDetector(config.identity), SFaceEncoder(config.identity)
        enrollment = Enrollment(config.identity)
        window = "Jake enrollment - one consenting resident only - q/Q cancels"
        try:
            with OpenCVCamera(config.pipeline.camera_id, config.camera) as camera:
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                for frame in camera:
                    faces = detector.detect(frame, BoundingBox(0, 0, 1, 1))
                    reason = "Exactly one face required; look forward then turn slightly"
                    if len(faces) == 1:
                        quality = face_quality(frame, faces[0], config.identity)
                        embedding = encoder.encode(frame, faces[0]) if quality.accepted else None
                        reason = enrollment.accept(quality, embedding, frame.captured_at)
                    display, _, _ = crop(frame, BoundingBox(0, 0, 1, 1))
                    text = (
                        f"{len(enrollment.samples)}/{config.identity.enrollment_samples} accepted "
                        f"| {reason}"
                    )
                    cv2.putText(
                        display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                    )
                    cv2.imshow(window, display)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q")):
                        print("Enrollment cancelled; no profile saved.")
                        return 0
                    if enrollment.ready:
                        store.add(enrollment.profile(name))
                        print(
                            f"Enrolled {name} from {len(enrollment.samples)} quality observations."
                        )
                        return 0
        finally:
            with suppress(cv2.error):
                cv2.destroyWindow(window)
        raise IdentityError("camera ended before enrollment completed; nothing saved")
    except KeyboardInterrupt:
        print("Enrollment cancelled; no partial profile saved.")
        return 0
    except Exception as exc:
        print(f"Resident enrollment error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
