"""Main-thread development rendering of the existing voice camera session."""

from contextlib import suppress
from datetime import UTC, datetime

import cv2
import numpy as np

from jake.application.voice_camera import VoicePreviewSnapshot
from jake.conversation import associate
from jake.preview import draw_people
from jake.voice_domain import VisualContext, VisualPerson

WINDOW = "Jake voice camera - q/Q to stop session"


def person_label(person: VisualPerson) -> str:
    if person.identity_state == "RESIDENT":
        return f"{person.display_name or 'Resident'} | RESIDENT"
    if person.identity_state != "UNKNOWN":
        return person.identity_state
    if person.visitor_state != "UNKNOWN":
        return f"VISITOR {(person.visitor_id or '')[:4].upper()} | {person.visitor_state}"
    return "UNKNOWN"


def context_label(context: VisualContext, at: datetime, max_age: float) -> str:
    person = associate(context, at, max_age)
    if person is None:
        return "VOICE CONTEXT: unresolved"
    return f"VOICE CONTEXT: {person_label(person)} | track={person.track_id}"


class VoicePreview:
    """Latest-frame display only: no capture, inference, audio work or persistence."""

    def __init__(self, max_age: float) -> None:
        self.max_age = max_age
        self._opened = False

    def update(self, snapshot: VoicePreviewSnapshot | None) -> bool:
        if not self._opened:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            self._opened = True
        if snapshot is not None:
            frame = snapshot.frame
            rgb = np.frombuffer(frame.pixels, dtype=np.uint8).reshape(frame.height, frame.width, 3)
            display = np.asarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), dtype=np.uint8)
            tracks = tuple(t for t in snapshot.tracks if t.visible)
            draw_people(display, tracks)
            people = {p.track_id: p for p in snapshot.context.people}
            for track in tracks:
                person = people.get(track.track_id)
                if person is not None:
                    cv2.putText(
                        display,
                        person_label(person),
                        (
                            min(frame.width - 1, int(track.box.left * frame.width)),
                            min(frame.height - 5, max(15, int(track.box.top * frame.height) + 20)),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        1,
                    )
            # A separate header keeps context readable even when a box meets the top edge.
            display = np.asarray(
                cv2.copyMakeBorder(display, 85, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)),
                dtype=np.uint8,
            )
            lines = (
                f"{frame.width}x{frame.height} | processing {snapshot.processing_fps:.1f} FPS "
                f"| seq {frame.sequence}",
                context_label(snapshot.context, datetime.now(UTC), self.max_age),
                "Visual context only - not acoustic speaker identification | q/Q quits",
            )
            for index, line in enumerate(lines):
                cv2.putText(
                    display,
                    line,
                    (10, 22 + index * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                )
            cv2.imshow(WINDOW, display)
        return cv2.waitKey(1) & 0xFF not in (ord("q"), ord("Q"))

    def close(self) -> None:
        if self._opened:
            with suppress(cv2.error):
                cv2.destroyWindow(WINDOW)
            self._opened = False
