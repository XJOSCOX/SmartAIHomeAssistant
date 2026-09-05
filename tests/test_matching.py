import pytest

from jake.domain import BoundingBox
from jake.matching import (
    assign_boxes,
    greedy_iou_assignment,
    hungarian_iou_assignment,
    intersection_over_union,
)


def test_exact_overlap() -> None:
    box = BoundingBox(0, 0, 0.5, 0.5)
    assert intersection_over_union(box, box) == 1.0


@pytest.mark.parametrize("left", [0.5, 0.6])
def test_no_overlap_including_touching_edges(left: float) -> None:
    assert intersection_over_union(BoundingBox(0, 0, 0.5, 0.5), BoundingBox(left, 0, 1, 1)) == 0


def test_partial_overlap_and_symmetry() -> None:
    first, second = BoundingBox(0, 0, 0.5, 0.5), BoundingBox(0.25, 0.25, 0.75, 0.75)
    assert intersection_over_union(first, second) == pytest.approx(1 / 7)
    assert intersection_over_union(second, first) == pytest.approx(1 / 7)


def test_tiny_box_underflow_is_safe() -> None:
    box = BoundingBox(0, 0, 1e-200, 1e-200)
    assert intersection_over_union(box, box) == 1
    assert intersection_over_union(box, BoundingBox(0, 0, 2e-200, 2e-200)) == 0


def test_highest_score_wins_over_track_order() -> None:
    first, second = BoundingBox(0, 0, 0.5, 0.5), BoundingBox(0.1, 0, 0.6, 0.5)
    assert greedy_iou_assignment((first, second), (second,), 0.3) == ((1, 0),)


def test_equal_scores_break_ties_by_track_then_detection_index() -> None:
    box = BoundingBox(0, 0, 1, 1)
    assert greedy_iou_assignment((box, box), (box, box), 0.3) == ((0, 0), (1, 1))
    assert greedy_iou_assignment((box, box), (box,), 0.3) == ((0, 0),)
    assert greedy_iou_assignment((box,), (box, box), 0.3) == ((0, 0),)


def test_threshold_is_inclusive_and_empty_inputs_are_safe() -> None:
    first, second = BoundingBox(0, 0, 0.5, 1), BoundingBox(0.25, 0, 0.75, 1)
    assert greedy_iou_assignment((first,), (second,), 1 / 3) == ((0, 0),)
    assert greedy_iou_assignment((first,), (second,), 0.34) == ()
    assert greedy_iou_assignment((), (second,), 0.3) == ()
    assert greedy_iou_assignment((first,), (), 0.3) == ()


def test_global_assignment_avoids_greedy_dead_end() -> None:
    tracks = (BoundingBox(0.2, 0, 0.4, 1), BoundingBox(0.3, 0, 0.5, 1))
    detections = (BoundingBox(0.22, 0, 0.42, 1), BoundingBox(0.1, 0, 0.3, 1))
    assert greedy_iou_assignment(tracks, detections, 0.3) == ((0, 0),)
    assert hungarian_iou_assignment(tracks, detections, 0.3) == ((0, 1), (1, 0))


@pytest.mark.parametrize("strategy", ["greedy", "hungarian"])
def test_assignment_gating_and_unmatched_cases(strategy: str) -> None:
    left, right = BoundingBox(0, 0, 0.3, 1), BoundingBox(0.7, 0, 1, 1)
    assert assign_boxes((left,), (right,), 0.3, strategy) == ()
    assert assign_boxes((left, right), (left,), 1.0, strategy) == ((0, 0),)
    assert assign_boxes((left,), (left, right), 1.0, strategy) == ((0, 0),)
    assert assign_boxes((), (left,), 0.3, strategy) == ()
    assert assign_boxes((left,), (), 0.3, strategy) == ()
    assert assign_boxes((left, left), (right, right), 0.3, strategy) == ()
    with pytest.raises(ValueError, match="unknown assignment"):
        assign_boxes((left,), (right,), 0.3, "bad")
