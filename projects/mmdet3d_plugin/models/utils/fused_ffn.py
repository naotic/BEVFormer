"""Fused Linear + ReLU for the mmcv FFN first sublayer (Python-only, no rebuild).

Gated by env ENABLE_FFN_GEMM_RELU=1 (default OFF -> baseline byte-identical).

mmcv FFN.layers = Sequential(
    Sequential(Linear(embed->ffn), ReLU(inplace), Dropout),  # <- fuse [Linear, ReLU]
    Linear(ffn->embed),
    Dropout)
The first Linear+ReLU is replaced by FusedLinearReLU, whose forward calls the
already-fused muDNN GEMM-RELU_BIAS epilogue (torch._addmm_activation, proven in
Stage 1: bit-exact, 2.13x) and whose backward is the standard relu-mask + 2 GEMMs.
Bit-exact (fwd+bwd) to nn.Linear + nn.ReLU.
"""
import os
import torch
from torch import nn


def ffn_gemm_relu_enabled():
    return os.environ.get("ENABLE_FFN_GEMM_RELU", "0").lower() in ("1", "true", "yes", "on")


class _FusedLinearReLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        # x: (..., in), weight: (out, in), bias: (out,)
        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1])
        # torch._addmm_activation(bias, mat1, mat2) = act(bias + mat1 @ mat2)
        # Linear computes x @ weight.t(); so mat2 = weight.t().
        out2d = torch._addmm_activation(bias, x2d, weight.t(), beta=1, alpha=1, use_gelu=False)
        out = out2d.reshape(*orig_shape[:-1], weight.shape[0])
        ctx.save_for_backward(x2d, weight, (out2d > 0))
        ctx.orig_shape = orig_shape
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x2d, weight, mask = ctx.saved_tensors
        go2d = grad_out.reshape(-1, weight.shape[0])
        grad_pre = go2d * mask  # relu backward (mask == out2d>0)

        grad_x = grad_w = grad_b = None
        if ctx.needs_input_grad[0]:
            grad_x = (grad_pre @ weight).reshape(ctx.orig_shape)
        if ctx.needs_input_grad[1]:
            grad_w = grad_pre.t() @ x2d
        if ctx.needs_input_grad[2]:
            grad_b = grad_pre.sum(0)
        return grad_x, grad_w, grad_b


class FusedLinearReLU(nn.Module):
    """Drop-in for Sequential(Linear, ReLU). Holds the same weight/bias params as
    the original Linear (rebound, not copied) so optimizer state & checkpoints are
    unaffected."""

    def __init__(self, linear):
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.in_features = linear.in_features
        self.out_features = linear.out_features

    def forward(self, x):
        if self.bias is None:
            # fused epilogue path needs a bias vector; fall back if absent
            return torch.relu(torch.nn.functional.linear(x, self.weight, None))
        return _FusedLinearReLUFn.apply(x, self.weight, self.bias)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, fused_relu=True"
