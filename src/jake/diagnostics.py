"""Development measurements independent of camera acquisition and model framework."""

from dataclasses import dataclass
from time import perf_counter

from jake.domain import Frame, PersonDetection
from jake.ports import PersonDetector


@dataclass(frozen=True, slots=True)
class DetectionMeasurement:
    detections: tuple[PersonDetection, ...]
    inference_ms: float

    @property
    def detection_fps(self) -> float:
        return 1000.0 / self.inference_ms if self.inference_ms > 0 else 0.0


def measure_detection(detector: PersonDetector, frame: Frame) -> DetectionMeasurement:
    """Time completed detect(): preprocessing, inference, and result conversion.

    Camera reads, drawing, and GUI waits are outside this measurement. This is
    detector-call latency, not just GPU kernel time. Initial warmup is included.
    """
    start = perf_counter()
    detections = detector.detect(frame)
    elapsed = perf_counter() - start
    return DetectionMeasurement(detections, elapsed * 1000.0)
