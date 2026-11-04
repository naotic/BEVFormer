"""Linear sum assignment helpers for DETR-style matchers.

Default path: per-matrix D2H + scipy; indices returned on the original device.
``linear_sum_assignment_batch`` stacks costs for a single D2H + parallel scipy.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import torch

from projects.mmdet3d_plugin.core.bbox.assigners.lapjv_torch import (
    hungarian_python,
    linear_sum_assignment_lapjv,
)

try:
    from scipy.optimize import linear_sum_assignment as scipy_linear_sum_assignment
except ImportError:
    scipy_linear_sum_assignment = None

_BATCH_POOL = ThreadPoolExecutor(max_workers=4)


def _solve_cpu(cost_np):
    if scipy_linear_sum_assignment is not None:
        return scipy_linear_sum_assignment(cost_np)
    return hungarian_python(cost_np.tolist())


@torch.no_grad()
def linear_sum_assignment(
        cost: torch.Tensor,
        *,
        use_device: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve LAP; semantics match ``scipy.optimize.linear_sum_assignment``."""
    if cost.ndim != 2:
        raise ValueError(f'cost must be 2-D, got shape {tuple(cost.shape)}')
    if cost.numel() == 0:
        empty = cost.new_zeros(0, dtype=torch.long)
        return empty, empty

    device = cost.device
    if use_device:
        return linear_sum_assignment_lapjv(cost)

    cost_cpu = cost.detach().cpu()
    if scipy_linear_sum_assignment is not None:
        row_inds, col_inds = scipy_linear_sum_assignment(cost_cpu)
        return (
            torch.as_tensor(row_inds, device=device, dtype=torch.long),
            torch.as_tensor(col_inds, device=device, dtype=torch.long),
        )

    row_inds, col_inds = hungarian_python(cost_cpu.numpy().tolist())
    return (
        torch.as_tensor(row_inds, device=device, dtype=torch.long),
        torch.as_tensor(col_inds, device=device, dtype=torch.long),
    )


@torch.no_grad()
def linear_sum_assignment_batch(
        costs: list[torch.Tensor]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Solve multiple LAPs with one stacked D2H + parallel scipy."""
    if not costs:
        return []
    if len(costs) == 1:
        r, c = linear_sum_assignment(costs[0])
        return [(r, c)]

    device = costs[0].device
    shapes = [tuple(c.shape) for c in costs]
    nrow = max(s[0] for s in shapes)
    ncol = max(s[1] for s in shapes)
    stacked = costs[0].new_full((len(costs), nrow, ncol), 1e6)
    for i, cost in enumerate(costs):
        h, w = cost.shape
        stacked[i, :h, :w] = cost.detach().float()

    stacked_cpu = stacked.cpu().numpy()
    pairs = list(_BATCH_POOL.map(
        lambda i: _solve_cpu(stacked_cpu[i, :shapes[i][0], :shapes[i][1]]),
        range(len(costs)),
    ))
    return [
        (
            torch.as_tensor(r, device=device, dtype=torch.long),
            torch.as_tensor(c, device=device, dtype=torch.long),
        )
        for r, c in pairs
    ]
