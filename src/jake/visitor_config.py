from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class VisitorConfig:
    enabled: bool = False
    required_observations: int = 5
    observation_window_seconds: float = 10.0
    observation_interval_seconds: float = 0.5
    carry_seconds: float = 2.0
    min_face_quality: float = 0.95
    match_similarity: float = 0.70
    ambiguity_margin: float = 0.10
    recurring_visit_count: int = 2
    retention_days: int = 30

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("visitors.enabled must be boolean")
        for name, minimum, maximum in (
            ("required_observations", 3, 20),
            ("recurring_visit_count", 2, 100),
            ("retention_days", 1, 365),
        ):
            v = getattr(self, name)
            if type(v) is not int or not minimum <= v <= maximum:
                raise ValueError(f"invalid visitors.{name}")
        for name in (
            "carry_seconds",
            "observation_window_seconds",
            "observation_interval_seconds",
            "min_face_quality",
            "match_similarity",
            "ambiguity_margin",
        ):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) or v <= 0:
                raise ValueError(f"invalid visitors.{name}")
        if (
            not 0 < self.min_face_quality <= 1
            or not 0 < self.ambiguity_margin < self.match_similarity <= 1
        ):
            raise ValueError("invalid visitor quality/similarity thresholds")
        if (
            self.required_observations - 1
        ) * self.observation_interval_seconds > self.observation_window_seconds:
            raise ValueError("visitor confirmation window too short")
