
# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform
import random
from functools import partial

import numpy as np
import torch
from mmcv.parallel import DataContainer, collate
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg
from torch.utils.data import DataLoader

from mmdet.datasets.samplers import GroupSampler
from projects.mmdet3d_plugin.datasets.samplers.group_sampler import DistributedGroupSampler
from projects.mmdet3d_plugin.datasets.samplers.distributed_sampler import DistributedSampler
from projects.mmdet3d_plugin.datasets.samplers.sampler import build_sampler

def pin_memory_recursive(obj, device=None):
    """Pin CPU tensors in nested mmcv batch structures.

    PyTorch's default DataLoader pin_memory does not recurse into
    DataContainer; BEVFormer batches use DataContainer(list[Tensor]).
    """
    if torch.is_tensor(obj):
        if obj.device.type == 'cpu':
            # Avoid re-pinning already pinned tensors (can be expensive).
            if hasattr(obj, "is_pinned") and obj.is_pinned():
                return obj
            if device:
                return obj.pin_memory(device)
            return obj.pin_memory()
        return obj

    if isinstance(obj, DataContainer):
        if obj.cpu_only:
            return obj
        return DataContainer(
            pin_memory_recursive(obj.data, device=device),
            stack=obj.stack,
            padding_value=obj.padding_value,
            cpu_only=obj.cpu_only,
            pad_dims=obj.pad_dims,
        )

    if isinstance(obj, dict):
        return {k: pin_memory_recursive(v, device=device) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        pinned = [pin_memory_recursive(v, device=device) for v in obj]
        return type(obj)(pinned)

    return obj


def _ensure_dataloader_pin_memory_patch():
    """Patch PyTorch pin thread to handle mmcv DataContainer batches."""
    import torch.utils.data._utils.pin_memory as pm

    if getattr(pm, '_bevformer_dc_patched', False):
        return

    _orig_pin_memory = pm.pin_memory

    def _pin_memory_with_datacontainer(data, device=None):
        if isinstance(data, DataContainer):
            return pin_memory_recursive(data, device=device)
        return _orig_pin_memory(data, device)

    pm.pin_memory = _pin_memory_with_datacontainer
    pm._bevformer_dc_patched = True


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     shuffler_sampler=None,
                     nonshuffler_sampler=None,
                     **kwargs):
    """Build PyTorch DataLoader.
    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.
    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        kwargs: any keyword argument to be used to initialize DataLoader
    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    if dist:
        # DistributedGroupSampler will definitely shuffle the data to satisfy
        # that images on each GPU are in the same group
        if shuffle:
            sampler = build_sampler(shuffler_sampler if shuffler_sampler is not None else dict(type='DistributedGroupSampler'),
                                     dict(
                                         dataset=dataset,
                                         samples_per_gpu=samples_per_gpu,
                                         num_replicas=world_size,
                                         rank=rank,
                                         seed=seed)
                                     )

        else:
            sampler = build_sampler(nonshuffler_sampler if nonshuffler_sampler is not None else dict(type='DistributedSampler'),
                                     dict(
                                         dataset=dataset,
                                         num_replicas=world_size,
                                         rank=rank,
                                         shuffle=shuffle,
                                         seed=seed)
                                     )

        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        # assert False, 'not support in bevformer'
        print('WARNING!!!!, Only can be used for obtain inference speed!!!!')
        sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    # DataLoader kwargs sanity:
    # - prefetch_factor is only valid when num_workers > 0
    # - persistent_workers requires num_workers > 0
    if num_workers == 0:
        kwargs.pop('prefetch_factor', None)
        kwargs.pop('persistent_workers', None)

    pin_memory = kwargs.pop('pin_memory', False)
    persistent_workers = kwargs.pop('persistent_workers', (num_workers > 0))

    if pin_memory:
        _ensure_dataloader_pin_memory_patch()

    collate_fn = partial(collate, samples_per_gpu=samples_per_gpu)

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        worker_init_fn=init_fn,
        persistent_workers=persistent_workers,
        **kwargs,
    )

    data_loader = DataLoader(**loader_kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Copyright (c) OpenMMLab. All rights reserved.
import platform
from mmcv.utils import Registry, build_from_cfg

from mmdet.datasets import DATASETS
from mmdet.datasets.builder import _concat_dataset

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

OBJECTSAMPLERS = Registry('Object sampler')


def custom_build_dataset(cfg, default_args=None):
    from mmdet3d.datasets.dataset_wrappers import CBGSDataset
    from mmdet.datasets.dataset_wrappers import (ClassBalancedDataset,
                                                 ConcatDataset, RepeatDataset)
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([custom_build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [custom_build_dataset(c, default_args) for c in cfg['datasets']],
            cfg.get('separate_eval', True))
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            custom_build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            custom_build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif cfg['type'] == 'CBGSDataset':
        dataset = CBGSDataset(custom_build_dataset(cfg['dataset'], default_args))
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset
