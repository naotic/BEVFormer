from .core.bbox.assigners.hungarian_assigner_3d import HungarianAssigner3D
from .core.bbox.coders.nms_free_coder import NMSFreeCoder
from .core.bbox.match_costs import BBox3DL1Cost
from .core.evaluation.eval_hooks import CustomDistEvalHook
from .datasets.pipelines import (
  PhotoMetricDistortionMultiViewImage, PadMultiViewImage,
  NormalizeMultiviewImage,  CustomCollect3D)
from .models.utils import *
from .models.opt.adamw import AdamW2
from .bevformer import *
if __import__("os").environ.get("BEVFORMER_SELECTED_CONFIG") != "1":
    from .dd3d import *


import os as _os2

from mmcv.cnn import CONV_LAYERS

if _os2.environ.get("BEVFORMER_STOCK_MMCV", "0").lower() in ("1", "true", "yes", "on"):
    print("[modulated_deform_conv] Export mode: keep stock mmcv DCNv2")
else:
    from modulated_deform_conv import ModulatedDeformConv2dPack as _OptDCNv2Pack
    from modulated_deform_conv.perf import patch_1x1_conv

    CONV_LAYERS.register_module("DCNv2", force=True, module=_OptDCNv2Pack)
    print("[modulated_deform_conv] Replaced mmcv DCNv2 with optimized implementation")
    patch_1x1_conv()

# FFN GEMM-RELU fusion (small, proven, Python-only, no rebuild). Replace the
# [Linear, ReLU] pair in each FFN's first sublayer with FusedLinearReLU, whose
# forward calls the Stage-1 fused muDNN GEMM-RELU_BIAS epilogue (torch.
# _addmm_activation) and whose backward is relu-mask + 2 GEMMs. Bit-exact (fwd+
# bwd) to Linear+ReLU. Gated by env ENABLE_FFN_GEMM_RELU=1, default OFF ->
# baseline byte-identical.
if _os2.environ.get("ENABLE_FFN_GEMM_RELU", "0").lower() in ("1", "true", "yes", "on"):
    import torch as _tff
    from torch import nn as _nnff
    from mmcv.cnn.bricks.transformer import FFN as _FFN2
    from .models.utils.fused_ffn import FusedLinearReLU as _FLR

    def _swap_ffn_linear_relu(ffn):
        # ffn.layers = Sequential( Sequential(Linear, ReLU, Dropout), ..., Linear, Dropout )
        layers = ffn.layers
        for sub in layers:
            if isinstance(sub, _tff.nn.Sequential) and len(sub) >= 2 \
                    and isinstance(sub[0], _nnff.Linear) \
                    and isinstance(sub[1], _nnff.ReLU):
                fused = _FLR(sub[0])
                sub[0] = fused
                sub[1] = _nnff.Identity()  # ReLU folded into fused[0]
        return ffn

    _orig_ffn_init = _FFN2.__init__
    def _ffn_init(self, *a, **kw):
        _orig_ffn_init(self, *a, **kw)
        # Only the ReLU-activated FFN is fusible; gelu/other acts fall through.
        act = getattr(self, "act_cfg", {}) or {}
        if act.get("type", "ReLU") == "ReLU":
            _swap_ffn_linear_relu(self)
    _FFN2.__init__ = _ffn_init
    print("[ffn_gemm_relu] FFN Linear+ReLU -> FusedLinearReLU (muDNN GEMM-RELU_BIAS epilogue)")

# CONV EPILOGUE (the BIG win): fuse backbone conv + FrozenBN + ReLU (+residual)
# into ONE muDNN Convolution::RunFusion kernel via
# torch.ops.musa_fused.conv2d_bias_act_add. Frozen BN's per-output-channel scale
# is folded into the conv weight in the forward and rescaled back in the backward,
# so the trainable weight grad is bit-exact. Skips the DCN conv2 (stages 3-4) and
# any conv whose following module is not FrozenBatchNorm2d. REQUIRES
# ENABLE_FROZEN_BN=1. Gated by env ENABLE_CONV_EPILOGUE=1, default OFF ->
# baseline byte-identical.
if _os2.environ.get("ENABLE_CONV_EPILOGUE", "0").lower() in ("1", "true", "yes", "on"):
    from .models.utils.conv_epilogue_patch import maybe_patch_bottleneck
    maybe_patch_bottleneck()
