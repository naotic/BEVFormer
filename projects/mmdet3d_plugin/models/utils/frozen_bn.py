"""Frozen BatchNorm for the image backbone — OPT-IN via env ENABLE_FROZEN_BN=1.

Replaces backbone BN with a fixed running-stat affine (the norm_eval=True intent;
norm_eval is not applied in this build, so backbone BN otherwise runs training-
mode). Forward is native eval-mode batch_norm (fast muDNN RunPure fwd + the
now-fixed torch_musa eval-BN backward), so no custom autograd is needed. Changes
BN semantics -> accuracy-gated (mAP validated: 0.0384 -> 0.0691).
"""
import os
import torch
import torch.nn.functional as F
from torch import nn
from mmcv.utils.parrots_wrapper import _BatchNorm


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d frozen to its running stats + affine. weight/bias/running_*
    are buffers (no grad); forward is native eval batch_norm, so grad to the input
    flows through the standard (now-correct on MUSA) eval-BN backward."""

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        # same buffer names as nn.BatchNorm2d so checkpoints load transparently
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var,
                            self.weight, self.bias, training=False, eps=self.eps)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # tolerate the num_batches_tracked key present in BN checkpoints
        num_batches_key = prefix + "num_batches_tracked"
        if num_batches_key in state_dict:
            del state_dict[num_batches_key]
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def extra_repr(self):
        return f"{self.num_features}, eps={self.eps}, frozen=True"


def convert_frozen_batchnorm(module):
    """Recursively replace _BatchNorm submodules with FrozenBatchNorm2d (in-place)."""
    if isinstance(module, _BatchNorm):
        frozen = FrozenBatchNorm2d(module.num_features, eps=module.eps)
        if module.affine and module.weight is not None:
            frozen.weight.data.copy_(module.weight.data)
            frozen.bias.data.copy_(module.bias.data)
        if module.running_mean is not None:
            frozen.running_mean.data.copy_(module.running_mean.data)
            frozen.running_var.data.copy_(module.running_var.data)
        return frozen
    for name, child in module.named_children():
        new_child = convert_frozen_batchnorm(child)
        if new_child is not child:
            setattr(module, name, new_child)
    return module


def maybe_freeze_backbone_bn(model, logger=None):
    """OPT-IN (ENABLE_FROZEN_BN=1): convert img_backbone BN -> FrozenBatchNorm2d.
    Changes BN semantics -> accuracy-gated; default OFF leaves baseline unchanged."""
    if os.environ.get("ENABLE_FROZEN_BN", "0") != "1":
        return model
    backbone = getattr(model, "img_backbone", None)
    if backbone is None:
        return model
    n_before = sum(1 for m in backbone.modules() if isinstance(m, _BatchNorm))
    convert_frozen_batchnorm(backbone)
    n_after = sum(1 for m in backbone.modules() if isinstance(m, FrozenBatchNorm2d))
    msg = (f"[frozen_bn] converted {n_after} backbone BatchNorm2d -> "
           f"FrozenBatchNorm2d (was {n_before} _BatchNorm)")
    (logger.info if logger else print)(msg)
    return model
