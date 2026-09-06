from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class VisitorConfig:
    enabled: bool = False
    required_observations: int = 5
    observation_window_seconds: float = 10.0
    observation_interval_seconds: float = 0.5
    carry_seconds: float = 2.0
    min_detector_confidence: float = 0.90
    resident_candidate_confirmations: int = 3
    resident_candidate_window_seconds: float = 3.0
    nonresident_recovery_observations: int = 3
    match_similarity: float = 0.70
    ambiguity_margin: float = 0.10
    recurrence_policy: str = "distinct_day"
    recurring_distinct_days: int = 2
    frequent_distinct_days: int = 5
    retention_days: int = 30

    def __post_init__(self) -> None:
        if self.recurrence_policy != "distinct_day":
            raise ValueError("visitors.recurrence_policy must be distinct_day")
        if type(self.enabled) is not bool:
            raise ValueError("visitors.enabled must be boolean")
        for name, minimum, maximum in (
            ("required_observations", 3, 20),
            ("recurring_distinct_days", 2, 100),
            ("frequent_distinct_days", 3, 365),
            ("retention_days", 1, 365),
            ("resident_candidate_confirmations", 2, 20),
            ("nonresident_recovery_observations", 2, 20),
        ):
            v = getattr(self, name)
            if type(v) is not int or not minimum <= v <= maximum:
                raise ValueError(f"invalid visitors.{name}")
        if self.frequent_distinct_days <= self.recurring_distinct_days:
            raise ValueError("frequent_distinct_days must exceed recurring_distinct_days")
        for name in (
            "carry_seconds",
            "observation_window_seconds",
            "observation_interval_seconds",
            "min_detector_confidence",
            "resident_candidate_window_seconds",
            "match_similarity",
            "ambiguity_margin",
        ):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) or v <= 0:
                raise ValueError(f"invalid visitors.{name}")
        if (
            not 0 < self.min_detector_confidence <= 1
            or not 0 < self.ambiguity_margin < self.match_similarity <= 1
        ):
            raise ValueError("invalid visitor quality/similarity thresholds")
        if (
            self.required_observations - 1
        ) * self.observation_interval_seconds > self.observation_window_seconds:
            raise ValueError("visitor confirmation window too short")
        if (
            self.resident_candidate_confirmations - 1
        ) * self.observation_interval_seconds > self.resident_candidate_window_seconds:
            raise ValueError("resident candidate window too short")
