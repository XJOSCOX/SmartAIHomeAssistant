"""Jake-owned dense linear assignment using Hungarian augmenting paths.

No SciPy or vision dependencies. Rectangles are padded to square with zero-cost
dummy cells. Row/column potentials and shortest augmenting paths yield O(n^3)
time and O(n^2) storage, n=max(rows, columns). Ordered scans settle ties.
"""

from collections.abc import Sequence
from math import inf, isfinite
from numbers import Real


class AssignmentError(ValueError):
    """A cost matrix is ragged, non-numeric, or non-finite."""


def linear_assignment(costs: Sequence[Sequence[float]]) -> tuple[tuple[int, int], ...]:
    """Minimum-cost assignment of min(rows, columns) pairs, sorted by row.

    Costs may be negative, but must be finite real numbers. This mathematical
    solver has no gating semantics; callers add unmatched nodes when necessary.
    Equal inputs yield identical assignments; ties are not identity information.
    """
    rows = len(costs)
    columns = len(costs[0]) if rows else 0
    for row in costs:
        if len(row) != columns:
            raise AssignmentError("cost matrix must be rectangular")
        for value in row:
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
                raise AssignmentError("costs must be finite real numbers")
    if not rows or not columns:
        return ()
    size = max(rows, columns)
    # Positive uniform scaling leaves the optimum unchanged and bounds potential
    # arithmetic for large finite costs. No cost-dependent epsilon changes ties.
    scale = max(1.0, max(abs(float(value)) for row in costs for value in row))
    matrix = [[0.0] * size for _ in range(size)]
    for i, row in enumerate(costs):
        for j, value in enumerate(row):
            matrix[i][j] = float(value) / scale

    row_potential = [0.0] * (size + 1)
    column_potential = [0.0] * (size + 1)
    owner = [0] * (size + 1)
    predecessor = [0] * (size + 1)
    for row_index in range(1, size + 1):
        owner[0] = row_index
        column = 0
        distance = [inf] * (size + 1)
        visited = [False] * (size + 1)
        while True:
            visited[column] = True
            active_row = owner[column]
            delta, next_column = inf, 0
            for candidate in range(1, size + 1):
                if not visited[candidate]:
                    reduced = (
                        matrix[active_row - 1][candidate - 1]
                        - row_potential[active_row]
                        - column_potential[candidate]
                    )
                    if reduced < distance[candidate]:
                        distance[candidate] = reduced
                        predecessor[candidate] = column
                    if distance[candidate] < delta:
                        delta, next_column = distance[candidate], candidate
            for candidate in range(size + 1):
                if visited[candidate]:
                    row_potential[owner[candidate]] += delta
                    column_potential[candidate] -= delta
                else:
                    distance[candidate] -= delta
            column = next_column
            if owner[column] == 0:
                break
        while column:
            previous = predecessor[column]
            owner[column] = owner[previous]
            column = previous
    return tuple(
        sorted(
            (owner[column] - 1, column - 1)
            for column in range(1, size + 1)
            if owner[column] <= rows and column <= columns
        )
    )
