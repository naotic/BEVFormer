"""Single-config optimization switches for BEVFormer-MUSA.

`cfg.bev_opt = dict(frozen_bn=True, conv_epilogue=True, ...)` in the config selects
every optimization; `apply_bev_opt(cfg)` maps it to the env vars the existing gated
code reads. Call it AFTER Config.fromfile and BEFORE the plugin import (so the
plugin's import-time monkeypatches and all later env-gated reads are config-driven).
No environment variables needed. Each rank sets its own env in-process.

(The DCN backend itself is specified directly in model.img_backbone.dcn.backend.)
"""
import os

_BEV_OPT_ENV = {
    # tf32 accuracy-safe stack (mAP-validated; recommended on)
    'frozen_bn':              'ENABLE_FROZEN_BN',         # backbone BN -> frozen affine
    'conv_epilogue':          'ENABLE_CONV_EPILOGUE',     # conv+BN+relu+add muDNN epilogue (needs frozen_bn)
    'conv_bn_fuse':           'ENABLE_CONV_BN_FUSE',       # downsample conv+BN (no relu) epilogue (needs conv_epilogue)
    'dcn_gemm_bn_relu':       'ENABLE_DCN_GEMM_BN_RELU',   # DCN main GEMM+BN+relu epilogue (needs conv_epilogue + fuse_c)
    'ffn_gemm_relu':          'ENABLE_FFN_GEMM_RELU',      # FFN Linear+ReLU GEMM epilogue
    'msda_col2im_smem':       'MSDA_COL2IM_SMEM_REDUCE',   # MSDA col2im backward smem-reduce (needs msda_musa_opt2)
    # prior backbone block
    'backbone_channels_last': 'ENABLE_BACKBONE_CL',        # backbone+neck NHWC
    'dcn_recompute_cols':     'DCN_RECOMPUTE_COLS',         # DCN bwd recompute Col (save mem)
    # (DCN backend itself is set directly in model.img_backbone.dcn.backend)
}


def apply_bev_opt(cfg):
    opt = cfg.get('bev_opt', None)
    if not opt:
        return
    applied = []
    for key, env in _BEV_OPT_ENV.items():
        if key in opt:
            val = opt[key]
            os.environ[env] = ('1' if val else '0') if isinstance(val, bool) else str(val)
            applied.append(f'{key}={val}')
    if applied:
        print('[bev_opt] config-driven optimizations: ' + ', '.join(applied))
