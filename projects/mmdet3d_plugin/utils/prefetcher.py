from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from mmcv.parallel import DataContainer
from projects.mmdet3d_plugin.datasets.builder import pin_memory_recursive


def _is_musa_available() -> bool:
    return hasattr(torch, "musa") and callable(getattr(torch.musa, "is_available", None)) and torch.musa.is_available()


def _musa_current_stream():
    return torch.musa.current_stream()


def _musa_stream(stream):
    return torch.musa.stream(stream)


def _musa_new_stream():
    return torch.musa.Stream()


def _to_musa(x: Any, non_blocking: bool) -> Any:
    if torch.is_tensor(x):
        if x.device.type == "musa":
            return x
        if x.device.type == "cpu":
            # For non_blocking H2D, the source CPU tensor should be pinned.
            # In this codebase, batch may be wrapped in mmcv.DataContainer,
            # so DataLoader's pin_memory=True might not cover all tensors.
            # if non_blocking and hasattr(x, "is_pinned") and hasattr(x, "pin_memory"):
            #     try:
            #         if not x.is_pinned():
            #             x = x.pin_memory()
            #     except RuntimeError:
            #         # Some tensors can't be pinned; fall back to regular transfer.
            #         pass
            if hasattr(x, "musa"):
                return x.musa(non_blocking=non_blocking)
            return x.to("musa", non_blocking=non_blocking)
        return x.to("musa", non_blocking=False)

    if isinstance(x, DataContainer):
        if x.cpu_only:
            return x
        return DataContainer(
            _to_musa(x.data, non_blocking=non_blocking),
            stack=x.stack,
            padding_value=x.padding_value,
            cpu_only=x.cpu_only,
            pad_dims=x.pad_dims,
        )

    if isinstance(x, Mapping):
        return {k: _to_musa(v, non_blocking=non_blocking) for k, v in x.items()}

    if isinstance(x, tuple):
        return tuple(_to_musa(v, non_blocking=non_blocking) for v in x)

    if isinstance(x, list):
        return [_to_musa(v, non_blocking=non_blocking) for v in x]

    if isinstance(x, np.ndarray):
        # Best-effort: keep dtype, avoid copy if possible.
        t = torch.from_numpy(x)
        return _to_musa(t, non_blocking=non_blocking)

    return x


def _record_stream(x: Any, stream) -> None:
    if torch.is_tensor(x):
        if x.device.type == "musa":
            x.record_stream(stream)
        return

    if isinstance(x, DataContainer):
        if x.cpu_only:
            return
        _record_stream(x.data, stream)
        return

    if isinstance(x, Mapping):
        for v in x.values():
            _record_stream(v, stream)
        return

    if isinstance(x, (tuple, list)):
        for v in x:
            _record_stream(v, stream)
        return


@dataclass(frozen=True)
class PrefetchConfig:
    non_blocking: bool = True
    # Pin on DataLoader's background thread (see builder._ensure_dataloader_pin_memory_patch).
    pin_cpu_batch: bool = False
    pin_keys: tuple[str, ...] = ("img",)


class PrefetchLoader:
    """GPU-side double-buffer prefetcher for MUSA backend.

    It overlaps host->device transfer with model compute by preloading next batch
    on a dedicated MUSA stream.
    """

    def __init__(self, loader: Iterable, *, config: PrefetchConfig | None = None):
        self.loader = loader
        self.config = config or PrefetchConfig()

        if not _is_musa_available():
            raise RuntimeError("torch.musa is not available, cannot enable MUSA prefetcher.")

    def __getattr__(self, name: str):
        # Make this wrapper transparent to MMCV hooks that expect a real
        # torch.utils.data.DataLoader (e.g. access to .sampler for set_epoch).
        return getattr(self.loader, name)

    def __len__(self) -> int:
        return len(self.loader)  # type: ignore[arg-type]

    def _prepare_cpu_batch(self, batch: Any) -> Any:
        if not self.config.pin_cpu_batch:
            return batch
        # Pin only selected keys to reduce CPU overhead.
        if isinstance(batch, dict):
            for k in self.config.pin_keys:
                if k in batch:
                    batch[k] = pin_memory_recursive(batch[k])
            return batch
        return pin_memory_recursive(batch)

    def __iter__(self) -> Iterator:
        it = iter(self.loader)
        stream = _musa_new_stream()
        current_stream = _musa_current_stream()

        try:
            first = self._prepare_cpu_batch(next(it))
        except StopIteration:
            return

        # import torch.distributed as dist
        # if (not dist.is_initialized() or dist.get_rank() == 0):
        #     t = first['img'].data
        #     if isinstance(t, list):
        #         t = t[0]
        #     print(f'[debug] before_prefetch: pinned={t.is_pinned()}, device={t.device}')

        with _musa_stream(stream):
            next_batch = _to_musa(first, non_blocking=self.config.non_blocking)

        while True:
            current_stream.wait_stream(stream)
            batch = next_batch
            _record_stream(batch, current_stream)

            try:
                nxt = self._prepare_cpu_batch(next(it))
            except StopIteration:
                yield batch
                break

            with _musa_stream(stream):
                next_batch = _to_musa(nxt, non_blocking=self.config.non_blocking)

            yield batch
