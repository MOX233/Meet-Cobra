"""Compiled transcription of alg_utils.km_algorithm, including tie order.

No fast-math, alternate assignment solver, or tolerance change is used.
This optional backend accelerates the same augmenting-path loop.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _augment(u, visited_x, visited_y, slack, slack_x, lx, ly, match_y, cost):
    visited_x[u] = True
    for v in range(cost.shape[0]):
        if visited_y[v]:
            continue
        gap = lx[u] + ly[v] - cost[u, v]
        if gap <= 1e-10:
            visited_y[v] = True
            if match_y[v] == -1 or _augment(match_y[v], visited_x, visited_y,
                                            slack, slack_x, lx, ly, match_y, cost):
                match_y[v] = u
                return True
        elif slack[v] > gap:
            slack[v] = gap
            slack_x[v] = u
    return False


@njit(cache=True)
def km_algorithm_compiled(cost_matrix):
    num_rows, num_cols = cost_matrix.shape
    n = max(num_rows, num_cols)
    if num_rows != num_cols:
        cost = -np.ones((n, n))
        cost[:num_rows, :num_cols] = cost_matrix
    else:
        cost = cost_matrix
    lx = np.empty(n)
    for i in range(n):
        lx[i] = np.max(cost[i])
    ly = np.zeros(n)
    match_y = -np.ones(n, dtype=np.int64)
    for u in range(n):
        slack = np.full(n, 1e9)
        slack_x = np.zeros(n, dtype=np.int64)
        while True:
            visited_x = np.zeros(n, dtype=np.bool_)
            visited_y = np.zeros(n, dtype=np.bool_)
            if _augment(u, visited_x, visited_y, slack, slack_x, lx, ly, match_y, cost):
                break
            delta = np.min(slack[~visited_y])
            for i in range(n):
                if visited_x[i]:
                    lx[i] -= delta
                if visited_y[i]:
                    ly[i] += delta
    matching = []
    total = 0.
    for v in range(num_cols):
        if match_y[v] != -1:
            matching.append((match_y[v], v))
            total += cost[match_y[v], v]
    return matching, total
