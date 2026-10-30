"""Fused conv + FrozenBN(affine) + ReLU(+residual) for the ResNet backbone.

THE BIG WIN (env ENABLE_CONV_EPILOGUE=1, REQUIRES ENABLE_FROZEN_BN=1).

The ResNet101 Bottleneck runs, on every backbone view (48/iter), the chain
    conv(bias=False) -> FrozenBN(y = scale*conv + shift) -> ReLU   [(+residual)]
which MUSA executes as a separate conv kernel, an affine kernel, a relu kernel
and (for conv3) a residual add kernel. Baseline relu_ ~157ms + add_ ~207ms are
mostly this. muDNN Convolution::RunFusion can do out = act(conv + bias + add) in
ONE kernel.

KEY MATH IDENTITY (bit-exact, NO training change). FrozenBN's scale/shift are
*frozen buffers*; only the conv weight w is trainable. Because conv is linear in
w and scale is per-OUTPUT-channel:

    relu( scale * conv(x, w) + shift + add )
      == relu( conv(x, w * scale[:,None,None,None]) + shift + add )

So in the FORWARD we fold the frozen scale into the weight
    w_folded = w * scale[:,None,None,None],   bias_folded = shift
and run the single fused conv epilogue. In the BACKWARD the trainable param is
still w; conv weight-grad is linear in the upstream grad and independent of the
weight value, so
    grad_w = conv_wgrad(x, grad_pre) * scale            (== unfused grad_w)
and scale/shift (frozen) get no grad. This is EXACT (fp32/tf32 noise only).

Gated by env ENABLE_CONV_EPILOGUE=1 AND ENABLE_FROZEN_BN=1. Default OFF ->
baseline byte-identical. Only the raw fused conv op lives in torch_musa
(torch.ops.musa_fused.conv2d_bias_act_add, default conv path untouched).
"""
import os
import torch
import bevformer_musa_fusion
from torch import nn
import torch.nn.functional as F


def conv_epilogue_enabled():
    return (
        os.environ.get("ENABLE_CONV_EPILOGUE", "0").lower() in ("1", "true", "yes", "on")
        and os.environ.get("ENABLE_FROZEN_BN", "0") == "1"
        and hasattr(torch.ops, "musa_fused")
    )


class FusedConvBNAct(torch.autograd.Function):
    """forward = relu?( conv(x, w*scale) + shift + add ), with frozen scale/shift.

    Saves x, w_folded, scale, and the relu mask. Backward reuses torch's
    convolution_backward on w_folded, then rescales grad_w by scale. No grad for
    the frozen scale/shift/add-as-buffer (add gets grad if it is a real tensor)."""

    @staticmethod
    def forward(ctx, x, weight, scale, shift, add, stride, padding, dilation,
                groups, act_relu):
        # scale/shift are frozen buffers (no grad). Fold scale into the weight.
        w_folded = weight * scale.reshape(-1, 1, 1, 1)
        bias_folded = shift  # length-Cout

        out = torch.ops.musa_fused.conv2d_bias_act_add(
            x, w_folded, bias_folded, add,
            list(stride), list(padding), list(dilation), int(groups),
            bool(act_relu),
        )

        ctx.act_relu = bool(act_relu)
        ctx.stride = list(stride)
        ctx.padding = list(padding)
        ctx.dilation = list(dilation)
        ctx.groups = int(groups)
        ctx.has_add = add is not None and add.numel() > 0
        # mask for relu backward; out>0 is exact (relu zeroes <=0)
        relu_mask = (out > 0) if act_relu else None
        ctx.save_for_backward(x, w_folded, scale, relu_mask)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, w_folded, scale, relu_mask = ctx.saved_tensors
        grad_pre = grad_out * relu_mask if ctx.act_relu else grad_out

        need_x = ctx.needs_input_grad[0]
        need_w = ctx.needs_input_grad[1]
        # residual add (input index 4) grad == grad_pre (identity wrt add)
        need_add = ctx.has_add and ctx.needs_input_grad[4]

        grad_x = grad_w_folded = None
        if need_x or need_w:
            gi, gw_folded, _ = torch.ops.aten.convolution_backward(
                grad_pre, x, w_folded, None,
                ctx.stride, ctx.padding, ctx.dilation, False, [0, 0], ctx.groups,
                [need_x, need_w, False],
            )
            grad_x = gi if need_x else None
            grad_w_folded = gw_folded if need_w else None

        grad_w = None
        if need_w:
            # dL/dw = dL/dw_folded * scale (per output channel)
            grad_w = grad_w_folded * scale.reshape(-1, 1, 1, 1)

        grad_add = grad_pre if need_add else None

        # order: x, weight, scale, shift, add, stride, padding, dilation, groups, act_relu
        return grad_x, grad_w, None, None, grad_add, None, None, None, None, None

@torch.compile(dynamic=True)
def fused_conv_bn_relu(x, conv, frozen_bn, add=None, act_relu=True):
    """Run conv (bias=False) -> frozen_bn affine -> relu (+optional residual add)
    as one fused muDNN kernel. `conv` is an nn.Conv2d (bias must be None) and
    `frozen_bn` is a FrozenBatchNorm2d with buffers weight/bias/running_mean/
    running_var/eps. Returns the same tensor as the unfused chain (bit-exact)."""
    # FrozenBN affine: y = scale*conv + shift, with
    #   scale = weight / sqrt(running_var + eps),  shift = bias - running_mean*scale
    scale = frozen_bn.weight * torch.rsqrt(frozen_bn.running_var + frozen_bn.eps)
    shift = frozen_bn.bias - frozen_bn.running_mean * scale
    stride = conv.stride
    padding = conv.padding
    dilation = conv.dilation
    groups = conv.groups
    return FusedConvBNAct.apply(
        x, conv.weight, scale, shift, add, stride, padding, dilation, groups,
        act_relu,
    )
