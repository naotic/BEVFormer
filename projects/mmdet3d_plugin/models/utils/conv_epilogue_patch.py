"""Monkeypatch mmdet ResNet Bottleneck.forward AND ResNet.forward (stem) to use
the fused conv epilogue.

Gated by env ENABLE_CONV_EPILOGUE=1 (REQUIRES ENABLE_FROZEN_BN=1). Default OFF
-> baseline byte-identical (the original forwards are left in place).

Fuses, per Bottleneck:
  * conv1 -> FrozenBN(norm1) -> ReLU            (no residual)
  * conv2 -> FrozenBN(norm2) -> ReLU            (no residual; SKIPPED for DCN conv2)
  * conv3 -> FrozenBN(norm3) -> (+downsample identity) -> ReLU  (residual + relu)

Fuses, in the ResNet stem (the LARGEST-spatial conv+BN+relu, run once per view,
~48 views/iter):
  * conv1 (7x7 s2, bias=False) -> FrozenBN(norm1) -> ReLU      (no residual)
  then the original eager maxpool. Only the non-deep-stem path is fused (deep_stem
  uses an nn.Sequential and falls back to the original forward entirely).

A conv is fused ONLY when it is a plain nn.Conv2d (skips ModulatedDeformConv2dPack
DCN conv2 in stages 3-4) AND its following norm is a FrozenBatchNorm2d. Anything
else falls back to the original eager conv/bn/relu. The downsample branch
(conv->bn, no relu) is left eager and only feeds the residual add. with_plugins
and with_cp (checkpoint) Bottlenecks fall back to the original forward entirely.

NOTE on the stem under frozen_stages=1: conv1/norm1 are frozen (requires_grad=
False), so FusedConvBNAct's backward computes neither a weight-grad (frozen) nor
an input-grad (the image input has no grad) -> the stem params stay frozen and
the only autograd effect is producing the same activation that flows downstream.
"""
import os
import torch
from torch import nn

from .frozen_bn import FrozenBatchNorm2d
from .fused_conv_bn_act import fused_conv_bn_relu, conv_epilogue_enabled
from .fused_dcn_bn_act import (
    fused_dcn_bn_relu, is_fusec_dcn, dcn_gemm_bn_relu_enabled,
)


def _is_fusible(conv, norm):
    return isinstance(conv, nn.Conv2d) and isinstance(norm, FrozenBatchNorm2d)


def _conv_bn_fuse_enabled():
    """Optimization 1: fuse the Bottleneck downsample branch conv+FrozenBN (no
    ReLU) into one muDNN conv epilogue. Env ENABLE_CONV_BN_FUSE=1; rides on the
    conv-epilogue patch (the Bottleneck.forward must already be patched)."""
    return os.environ.get("ENABLE_CONV_BN_FUSE", "0").lower() in ("1", "true", "yes", "on")


def _fused_downsample(downsample, x):
    """Downsample identity = conv -> FrozenBN (NO relu, NO residual). When the
    branch is exactly nn.Sequential(nn.Conv2d, FrozenBatchNorm2d) and ENABLE_CONV_BN_FUSE
    is on, run it as ONE fused muDNN conv epilogue (act_relu=False, add=None) ==
    bit-exact to the eager conv->bn. Anything else (avg_down pooling, non-frozen
    norm, etc.) falls back to the original eager downsample."""
    if (_conv_bn_fuse_enabled()
            and isinstance(downsample, nn.Sequential) and len(downsample) == 2
            and _is_fusible(downsample[0], downsample[1])
            and downsample[0].bias is None):
        return fused_conv_bn_relu(x, downsample[0], downsample[1], add=None, act_relu=False)
    return downsample(x)


def make_fused_bottleneck_forward(orig_forward):
    def forward(self, x):
        # Fall back to original for any non-trivial path we don't fuse.
        if getattr(self, "with_plugins", False):
            return orig_forward(self, x)
        if getattr(self, "with_cp", False) and x.requires_grad:
            return orig_forward(self, x)

        n1, n2, n3 = self.norm1, self.norm2, self.norm3
        c1, c2, c3 = self.conv1, self.conv2, self.conv3

        # conv1 -> bn1 -> relu
        if _is_fusible(c1, n1):
            out = fused_conv_bn_relu(x, c1, n1, add=None, act_relu=True)
        else:
            out = self.relu(n1(c1(x)))

        # conv2 -> bn2 -> relu. Plain conv2 -> conv epilogue. DCN conv2 (stages 3-4,
        # not an nn.Conv2d) -> optional DCN GEMM+FrozenBN+ReLU epilogue (opt 2);
        # else the eager relu(bn(dcn(x))).
        if _is_fusible(c2, n2):
            out = fused_conv_bn_relu(out, c2, n2, add=None, act_relu=True)
        elif (dcn_gemm_bn_relu_enabled() and is_fusec_dcn(c2)
                and isinstance(n2, FrozenBatchNorm2d)):
            out = fused_dcn_bn_relu(out, c2, n2, act_relu=True)
        else:
            out = self.relu(n2(c2(out)))

        # downsample identity (conv->bn, no relu); fused when ENABLE_CONV_BN_FUSE=1
        identity = _fused_downsample(self.downsample, x) if self.downsample is not None else x

        # conv3 -> bn3 -> (+identity) -> relu  (fused residual add + relu)
        if _is_fusible(c3, n3):
            out = fused_conv_bn_relu(out, c3, n3, add=identity, act_relu=True)
        else:
            out = n3(c3(out))
            out = out + identity
            out = self.relu(out)
        return out

    return forward


def make_fused_resnet_forward(orig_forward):
    """Wrap mmdet ResNet.forward so the (non-deep-stem) stem chain
    conv1 -> FrozenBN(norm1) -> ReLU is fused into one muDNN conv epilogue.
    The eager maxpool and the res_layers loop are reproduced exactly; the
    deep_stem path (nn.Sequential) is left to the original forward."""
    def forward(self, x):
        # deep_stem path is an nn.Sequential -> not fused here; fall back.
        if getattr(self, "deep_stem", False):
            return orig_forward(self, x)

        c1, n1 = self.conv1, self.norm1
        if _is_fusible(c1, n1):
            x = fused_conv_bn_relu(x, c1, n1, add=None, act_relu=True)
        else:
            x = self.relu(n1(c1(x)))

        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    return forward


_patched = False


def maybe_patch_bottleneck(logger=None):
    """Install the fused Bottleneck.forward when ENABLE_CONV_EPILOGUE=1 (and
    ENABLE_FROZEN_BN=1). Idempotent; no-op otherwise."""
    global _patched
    if _patched:
        return
    if os.environ.get("ENABLE_CONV_EPILOGUE", "0").lower() not in ("1", "true", "yes", "on"):
        return
    if os.environ.get("ENABLE_FROZEN_BN", "0") != "1":
        msg = ("[conv_epilogue][WARN] ENABLE_CONV_EPILOGUE=1 but ENABLE_FROZEN_BN!=1; "
               "the fused conv epilogue requires frozen BN. NOT patching.")
        (logger.warning if logger else print)(msg)
        return
    if not hasattr(torch.ops, "musa_fused"):
        (logger.warning if logger else print)(
            "[conv_epilogue][WARN] torch.ops.musa_fused.conv2d_bias_act_add unavailable "
            "(torch_musa not rebuilt with the fused conv op). NOT patching.")
        return

    import mmdet.models.backbones.resnet as _R
    _R.Bottleneck.forward = make_fused_bottleneck_forward(_R.Bottleneck.forward)
    _R.ResNet.forward = make_fused_resnet_forward(_R.ResNet.forward)
    _patched = True
    (logger.info if logger else print)(
        "[conv_epilogue] ResNet Bottleneck.forward + ResNet.forward (stem) patched "
        "-> fused conv+FrozenBN+ReLU(+residual) via "
        "torch.ops.musa_fused.conv2d_bias_act_add")
