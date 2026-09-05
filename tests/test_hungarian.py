from collections.abc import Sequence
from itertools import permutations
from random import Random
from typing import cast

import pytest

from jake.hungarian import AssignmentError, linear_assignment


@pytest.mark.parametrize(
    "matrix,expected",
    [
        ([[5.0]], ((0, 0),)),
        ([[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]], ((0, 1), (1, 0), (2, 2))),
        ([[1.0, 2.0], [2.0, 100.0]], ((0, 1), (1, 0))),  # Greedy costs 101; global costs 4.
        ([[5.0, 1.0, 9.0], [1.0, 8.0, 7.0]], ((0, 1), (1, 0))),
        ([[9.0, 9.0], [1.0, 8.0], [8.0, 1.0]], ((1, 0), (2, 1))),
        ([[-5.0, -1.0], [-2.0, -4.0]], ((0, 0), (1, 1))),
        ([[1e308, -1e308], [-1e308, 1e308]], ((0, 1), (1, 0))),
        ([], ()),
        ([[], []], ()),
    ],
)
def test_known_optima(matrix: list[list[float]], expected: tuple[tuple[int, int], ...]) -> None:
    assert linear_assignment(matrix) == expected


def test_deterministic_ties_and_input_unchanged() -> None:
    matrix = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    for _ in range(10):
        assert linear_assignment(matrix) == ((0, 0), (1, 1))
    assert matrix == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]


@pytest.mark.parametrize(
    "matrix",
    [[[1], [2, 3]], [[float("nan")]], [[float("inf")]], [[-float("inf")]], [[True]], [["1"]]],
)
def test_invalid_costs(matrix: object) -> None:
    with pytest.raises(AssignmentError):
        linear_assignment(cast(Sequence[Sequence[float]], matrix))


def test_small_rectangles_match_exhaustive_optimum() -> None:
    random = Random(711)
    for rows in range(1, 5):
        for columns in range(1, 5):
            for _ in range(8):
                matrix = [
                    [float(random.randrange(-5, 10)) for _ in range(columns)] for _ in range(rows)
                ]
                result = linear_assignment(matrix)
                assert len(result) == min(rows, columns)
                assert len({i for i, _ in result}) == len(result)
                assert len({j for _, j in result}) == len(result)
                if rows <= columns:
                    optimum = min(
                        sum(matrix[i][j] for i, j in enumerate(p))
                        for p in permutations(range(columns), rows)
                    )
                else:
                    optimum = min(
                        sum(matrix[i][j] for j, i in enumerate(p))
                        for p in permutations(range(rows), columns)
                    )
                assert sum(matrix[i][j] for i, j in result) == optimum
