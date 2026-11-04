"""LAP helpers for Hungarian matching (P2).

Rectangular BEVFormer costs use scipy JV (Jonker-Volgenant) on a single CPU
view; row/col indices are materialized on the original device.  A pure-Python
JV fallback exists when scipy is unavailable.
"""

from __future__ import annotations

import torch

try:
    from scipy.optimize import linear_sum_assignment as scipy_lsa
except ImportError:
    scipy_lsa = None


def hungarian_python(cost_rows: list[list[float]]) -> tuple[list[int], list[int]]:
    """Pure-Python JV for rectangular matrices (``n <= m`` after transpose)."""
    cost = [list(row) for row in cost_rows]
    n, m = len(cost), len(cost[0])
    transposed = False
    if n > m:
        cost = [list(x) for x in zip(*cost)]
        n, m = len(cost), len(cost[0])
        transposed = True

    u = [0] * (n + 1)
    v = [0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float('inf')] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float('inf')
            j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    row_inds, col_inds = [], []
    for j in range(1, m + 1):
        if p[j] != 0:
            row_inds.append(p[j] - 1)
            col_inds.append(j - 1)
    if transposed:
        row_inds, col_inds = col_inds, row_inds
    return row_inds, col_inds


def linear_sum_assignment_lapjv(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve LAP; indices returned on ``cost.device`` (scipy JV when available)."""
    if cost.ndim != 2:
        raise ValueError(f'cost must be 2-D, got {tuple(cost.shape)}')
    if cost.numel() == 0:
        empty = cost.new_zeros(0, dtype=torch.long)
        return empty, empty

    nrow, ncol = cost.shape
    if nrow == 0 or ncol == 0:
        empty = cost.new_zeros(0, dtype=torch.long)
        return empty, empty

    device = cost.device
    cost_cpu = cost.detach().float().cpu()
    if scipy_lsa is not None:
        row_inds, col_inds = scipy_lsa(cost_cpu.numpy())
    else:
        row_inds, col_inds = hungarian_python(cost_cpu.numpy().tolist())
    return (
        torch.as_tensor(row_inds, device=device, dtype=torch.long),
        torch.as_tensor(col_inds, device=device, dtype=torch.long),
    )
