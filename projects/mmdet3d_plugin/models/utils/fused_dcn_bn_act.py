"""Optimization 2: fuse the DCNv2 main GEMM + FrozenBN + ReLU.

In the ResNet stages-3/4 Bottleneck, conv2 is a ModulatedDeformConv2dPack (fuse_c
backend) followed by FrozenBN(norm2) + ReLU. The fuse_c DCN ends in a muBLAS main
GEMM (col @ w_flat); the eager chain then runs a separate batch_norm kernel and a
relu kernel on its output (the trace's ModulatedDeformConv2dPack -> FrozenBatchNorm2d
-> ReLU box).

KEY MATH IDENTITY (bit-exact, NO training change). FrozenBN's scale/shift are
frozen per-OUTPUT-channel buffers; only the DCN weight is trainable. The GEMM is
linear in the weight, so
    relu( scale * (col @ w_flat + bias) + shift )
      == relu( col @ (w_flat * scale) + (shift + scale*bias) )
which muBLAS runs as ONE GEMM-BIAS-RELU epilogue (torch._addmm_activation). In the
backward the trainable param is still the DCN weight; folding relu_mask*scale into
the upstream GEMM grad makes the weight/bias/col grads identical to the unfused
chain (tf32/fp noise only). The fold lives in DCNSplitNHWCFunction (bev_kernels
dcn_split_nhwc.py); here we just compute scale/shift from the FrozenBN and route
the DCN through its epilogue path.

Gated by env ENABLE_DCN_GEMM_BN_RELU=1 (rides on the conv-epilogue Bottleneck
patch; REQUIRES ENABLE_FROZEN_BN=1 and the DCN fuse_c backend). Default OFF ->
baseline byte-identical (the eager relu(bn(dcn(x))) is used).
"""
import os
import torch

from .frozen_bn import FrozenBatchNorm2d


def dcn_gemm_bn_relu_enabled():
    return (
        os.environ.get("ENABLE_DCN_GEMM_BN_RELU", "0").lower() in ("1", "true", "yes", "on")
        and os.environ.get("ENABLE_FROZEN_BN", "0") == "1"
    )


def is_fusec_dcn(conv):
    """A fuse_c ModulatedDeformConv2dPack the epilogue path supports (groups==1,
    deform_groups==1; the fuse_c constraints)."""
    return (
        getattr(conv, "backend", None) == "fuse_c"
        and hasattr(conv, "conv_offset")
        and getattr(conv, "groups", 1) == 1
        and getattr(conv, "deform_groups", 1) == 1
    )


def fused_dcn_bn_relu(x, dcn, frozen_bn, act_relu=True):
    """Run the DCNv2 (fuse_c) main GEMM with a fused FrozenBN(affine)+ReLU
    epilogue == relu( frozen_bn( dcn(x) ) ), bit-exact to the eager chain. `dcn`
    is a ModulatedDeformConv2dPack (fuse_c); `frozen_bn` a FrozenBatchNorm2d."""
    # FrozenBN affine: y = scale*z + shift, scale = w/sqrt(var+eps), shift = b - mean*scale
    scale = frozen_bn.weight * torch.rsqrt(frozen_bn.running_var + frozen_bn.eps)
    shift = frozen_bn.bias - frozen_bn.running_mean * scale
    from modulated_deform_conv.dcn_split_nhwc import dcn_apply_nchw
    return dcn_apply_nchw(
        x, dcn.conv_offset.weight, dcn.conv_offset.bias, dcn.weight, dcn.bias,
        dcn.stride, dcn.padding, dcn.dilation,
        bn_scale=scale, bn_shift=shift, act_relu=act_relu,
    )
