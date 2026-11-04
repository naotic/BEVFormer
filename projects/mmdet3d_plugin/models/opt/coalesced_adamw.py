"""Batch AdamW updates with equal options while preserving checkpoint groups."""
import torch
from mmcv.runner.optimizer.builder import OPTIMIZERS


@OPTIMIZERS.register_module()
class CoalescedAdamW(torch.optim.AdamW):
    def step(self, closure=None):
        adamw_step = torch.optim.AdamW.step
        # The outer optimizer call owns hooks; avoid invoking them twice.
        if getattr(adamw_step, 'hooked', False):
            adamw_step = adamw_step.__wrapped__
        if closure is not None:
            return adamw_step(self, closure)

        original_groups = self.param_groups
        merged = {}
        for group in original_groups:
            # Tensor options group by identity without synchronizing the device.
            key = tuple(
                (name, ('tensor', id(value)) if isinstance(value, torch.Tensor)
                 else value)
                for name, value in sorted(group.items())
                if name not in ('params', 'param_names'))
            try:
                hash(key)
            except TypeError:
                key = ('unmergeable', id(group))
            if key not in merged:
                merged[key] = {name: value for name, value in group.items()
                               if name not in ('params', 'param_names')}
                merged[key]['params'] = []
            merged[key]['params'].extend(group['params'])

        self.param_groups = list(merged.values())
        try:
            return adamw_step(self, closure)
        finally:
            # Keep scheduler indices and existing optimizer checkpoints valid.
            self.param_groups = original_groups
