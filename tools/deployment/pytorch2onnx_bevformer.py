#!/usr/bin/env python3
# Copyright (c) OpenMMLab / local BEVFormer-MUSA fork.
"""Export BEVFormer to ONNX and report adaptation blockers.

Even when export fails, this script walks stage-by-stage (load -> eager
forward -> submodule export -> full export) and prints a concrete adaptation
checklist.

Example:
  cd /path/to/BEVFormer
  PYTHONPATH=. python tools/deployment/pytorch2onnx_bevformer.py \\
    projects/configs/bevformer/bevformer_base.py \\
    ckpts/bevformer_r101_dcn_24ep.pth \\
    --out work_dirs/bevformer_base.onnx \\
    --device musa:0
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple

# Thread caps (match train/test scripts)
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

# Export must instantiate MMCV's DCNv2. The optimized MUSA registry replacement
# uses a custom autograd.Function that the legacy ONNX exporter cannot lower.
if os.environ.get('BEVFORMER_KEEP_BEV_OPT', '0') != '1':
    os.environ['BEVFORMER_STOCK_MMCV'] = '1'
# Disable MUSA training fusions that are not ONNX-friendly.
if os.environ.get('BEVFORMER_KEEP_BEV_OPT', '0') != '1':
    for _k in (
            'ENABLE_FROZEN_BN', 'ENABLE_CONV_EPILOGUE', 'ENABLE_CONV_BN_FUSE',
            'ENABLE_DCN_GEMM_BN_RELU', 'ENABLE_FFN_GEMM_RELU',
            'MSDA_COL2IM_SMEM_REDUCE', 'ENABLE_BACKBONE_CL', 'DCN_RECOMPUTE_COLS'):
        os.environ[_k] = '0'
# Avoid torch.compile / dynamo wrapping methods during export probing.
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import cv2  # noqa: E402
cv2.setNumThreads(0)

import torch  # noqa: E402
torch.set_num_threads(1)
try:
    import torch._dynamo as _dynamo
    _dynamo.config.disable = True
except Exception:
    pass

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TOOLS_DIR = os.path.join(REPO_ROOT, 'tools')
for _p in (REPO_ROOT, TOOLS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    stage: str
    status: str  # ok / fail / warn / info
    message: str
    adapt: str = ''


@dataclass
class Report:
    findings: List[Finding] = field(default_factory=list)

    def add(self, stage: str, status: str, message: str, adapt: str = '') -> None:
        self.findings.append(Finding(stage, status, message, adapt))
        tag = status.upper()
        print(f'[{tag}][{stage}] {message}')
        if adapt:
            print(f'         -> adapt: {adapt}')

    def dump(self, path: Optional[str] = None) -> str:
        lines = ['# BEVFormer -> ONNX adaptation report', '']
        adapts = []
        for f in self.findings:
            lines.append(f'- **{f.status.upper()}** `{f.stage}`: {f.message}')
            if f.adapt:
                lines.append(f'  - adapt: {f.adapt}')
                adapts.append(f.adapt)
        lines.append('')
        lines.append('## Deduped adaptation checklist')
        lines.append('')
        seen = set()
        idx = 1
        for a in adapts:
            if a not in seen:
                seen.add(a)
                lines.append(f'{idx}. {a}')
                idx += 1
        if not adapts:
            lines.append('1. (no blockers recorded)')
        text = '\n'.join(lines) + '\n'
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
            with open(path, 'w', encoding='utf-8') as fp:
                fp.write(text)
            print(f'[INFO] wrote report: {path}')
        return text


# Known static adaptation items (always printed; runtime may add more).
STATIC_ADAPTATIONS = [
    ('Python bool × Tensor in get_bev_features',
     'In PerceptionTransformer.get_bev_features, replace '
     '`shift_y * self.use_shift` / `can_bus * self.use_can_bus` with '
     'tensor scalars (e.g. shift_y * float(self.use_shift)); tracer emits '
     'aten::mul(Tensor, bool) which ONNX cannot lower.'),
    ('img_metas dict / Python control flow',
     'Replace img_metas Python dict with Tensor inputs '
     '(can_bus[B,18], lidar2img[B,N,4,4], img_hw[2]); rewrite '
     'PerceptionTransformer.get_bev_features / BEVFormerEncoder.point_sampling '
     'to use pure torch ops (drop np.array / list comprehensions).'),
    ('extract_img_feat squeeze when B==1',
     'Remove or gate `img.squeeze_()` in BEVFormer.extract_img_feat; keep '
     'explicit [B,N,C,H,W] -> [B*N,C,H,W] reshape for export-safe batch dims.'),
    ('Temporal state outside graph',
     'Export single-frame graph with prev_bev as an explicit I/O tensor; '
     'keep scene_token / ego delta bookkeeping in host code '
     '(do not export forward_test state machine).'),
    ('DCNv2 custom op',
     'Provide ONNX symbolic for DCNv2, or replace backbone DCN stages with '
     'plain Conv2d for export, or export backbone separately with a '
     'custom ORT/TensorRT plugin.'),
    ('MSDeformableAttention custom op',
     'Force multi_scale_deformable_attn_pytorch (grid_sample path) for export, '
     'or register ORT/TRT plugin for ms_deform_attn_forward.'),
    ('torchvision.transforms.functional.rotate on prev_bev',
     'Rewrite rotate_prev_bev with torch grid_sample / affine_grid '
     '(no Python for-loop over batch).'),
    ('Post-processing / NMSFreeCoder',
     'Export raw heads only (cls/bbox/bev_embed); run get_bboxes on host '
     '(uses box_type_3d Python objects).'),
    ('MUSA-only fused kernels (bev_opt)',
     'Disable bev_opt / BEVFORMER_STOCK_MMCV=1 for export; do not rely on '
     'fuse_c DCN, conv_epilogue, ffn_gemm_relu, channels_last.'),
    ('@torch.compile on BEVFormerHead.post_precess',
     'Disable torch.compile / call the underlying eager function during export.'),
]


# ---------------------------------------------------------------------------
# Export-only operator replacements
# ---------------------------------------------------------------------------

def _pair(v):
    if isinstance(v, tuple):
        return v
    return (int(v), int(v))


def rotate_prev_bev_export(prev_bev: torch.Tensor, angle_deg: torch.Tensor,
                           bev_h: int, bev_w: int) -> torch.Tensor:
    """Match torchvision.rotate(nearest, center=image center) with tensors."""
    num_query, batch, channels = prev_bev.shape
    image = prev_bev.permute(1, 2, 0).reshape(
        batch, channels, int(bev_h), int(bev_w))
    angle = angle_deg.to(device=image.device, dtype=image.dtype)
    radians = angle * (torch.pi / 180.0)
    cos_a = torch.cos(radians).view(batch, 1, 1)
    sin_a = torch.sin(radians).view(batch, 1, 1)

    # Pixel-center coordinates and normalization mirror torchvision's
    # _gen_affine_grid for expand=False and align_corners=False.
    x = torch.linspace(
        -float(bev_w) * 0.5 + 0.5, float(bev_w) * 0.5 - 0.5,
        steps=int(bev_w), device=image.device, dtype=image.dtype)
    y = torch.linspace(
        -float(bev_h) * 0.5 + 0.5, float(bev_h) * 0.5 - 0.5,
        steps=int(bev_h), device=image.device, dtype=image.dtype)
    x = x.view(1, 1, int(bev_w)).expand(batch, int(bev_h), int(bev_w))
    y = y.view(1, int(bev_h), 1).expand(batch, int(bev_h), int(bev_w))
    grid_x = (cos_a * x - sin_a * y) / (float(bev_w) * 0.5)
    grid_y = (sin_a * x + cos_a * y) / (float(bev_h) * 0.5)
    grid = torch.stack((grid_x, grid_y), dim=-1)
    rotated = torch.nn.functional.grid_sample(
        image, grid, mode='nearest', padding_mode='zeros',
        align_corners=False)
    return rotated.reshape(batch, channels, num_query).permute(2, 0, 1)


def dcnv2_grid_sample(input: torch.Tensor, offset: torch.Tensor,
                      mask: torch.Tensor, weight: torch.Tensor,
                      bias: Optional[torch.Tensor], stride, padding, dilation,
                      groups: int, deform_groups: int) -> torch.Tensor:
    """ModulatedDeformConv2d via standard torch ops for ONNX export."""
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)
    bsz, in_channels, in_h, in_w = input.shape
    out_channels, channels_per_group, k_h, k_w = weight.shape
    _, _, out_h, out_w = offset.shape
    kernel = k_h * k_w

    if groups != 1:
        raise RuntimeError('export DCNv2 grid_sample path currently expects groups=1')
    if in_channels != channels_per_group:
        raise RuntimeError('unexpected DCNv2 weight/input channel layout')

    dtype = input.dtype
    device = input.device
    yy = torch.arange(out_h, device=device, dtype=dtype) * float(stride_h) - float(pad_h)
    xx = torch.arange(out_w, device=device, dtype=dtype) * float(stride_w) - float(pad_w)
    base_y, base_x = torch.meshgrid(yy, xx, indexing='ij')

    sampled = []
    for kh in range(k_h):
        for kw in range(k_w):
            k = kh * k_w + kw
            dg = min(k * deform_groups // kernel, deform_groups - 1)
            off_base = dg * 2 * kernel + 2 * (k - dg * kernel // deform_groups)
            y = base_y + float(kh * dil_h) + offset[:, off_base, :, :]
            x = base_x + float(kw * dil_w) + offset[:, off_base + 1, :, :]
            y_norm = y * (2.0 / max(in_h - 1, 1)) - 1.0
            x_norm = x * (2.0 / max(in_w - 1, 1)) - 1.0
            grid = torch.stack((x_norm, y_norm), dim=-1)
            sampled_k = torch.nn.functional.grid_sample(
                input, grid, mode='bilinear', padding_mode='zeros',
                align_corners=True)
            sampled.append(sampled_k * mask[:, k:k + 1, :, :])

    cols = torch.stack(sampled, dim=2)  # [B, Cin, K, OH, OW]
    weight_flat = weight.reshape(out_channels, in_channels, kernel)
    out = torch.einsum('bckhw,ock->bohw', cols, weight_flat)
    if bias is not None:
        out = out + bias.reshape(1, -1, 1, 1)
    return out


def patch_export_only_ops(report: Report) -> None:
    """Patch training-only/custom ops to ONNX-supported torch ops in this process."""
    try:
        from mmcv.cnn import CONV_LAYERS
        from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2dPack
    except Exception as e:
        report.add('export.patch', 'warn', f'DCNv2 patch skipped: {e}')
        return

    # Restore stock DCNv2 even when the plugin registered an optimized class.
    if os.environ.get('BEVFORMER_KEEP_NATIVE_DCN', '0') != '1':
        CONV_LAYERS.register_module(
            'DCNv2', force=True, module=ModulatedDeformConv2dPack)
        report.add('export.registry', 'ok',
                   'restored CONV_LAYERS[DCNv2] to stock mmcv class')

    if getattr(ModulatedDeformConv2dPack, '_bevformer_onnx_grid_patch', False):
        return

    def _forward_export(self, x):
        out = self.conv_offset(x)
        o1, o2, m = torch.chunk(out, 3, dim=1)
        off = torch.cat((o1, o2), dim=1)
        m = torch.sigmoid(m)
        return dcnv2_grid_sample(x, off, m, self.weight, self.bias,
                                 self.stride, self.padding, self.dilation,
                                 self.groups, self.deform_groups)

    if os.environ.get('BEVFORMER_KEEP_NATIVE_DCN', '0') != '1':
        ModulatedDeformConv2dPack.forward = _forward_export
        ModulatedDeformConv2dPack._bevformer_onnx_grid_patch = True
    os.environ['BEVFORMER_ONNX_EXPORT_DCN_GRID'] = '1'
    report.add('export.patch', 'ok',
               'patched ModulatedDeformConv2dPack.forward to grid_sample path',
               'Target runtime must support ONNX GridSample (opset >= 16/17).')

    try:
        from projects.mmdet3d_plugin.bevformer.modules.transformer import PerceptionTransformer
        from projects.mmdet3d_plugin.bevformer.modules.encoder import BEVFormerEncoder
        from projects.mmdet3d_plugin.bevformer.modules.spatial_cross_attention import SpatialCrossAttention
    except Exception as e:
        report.add('export.patch', 'warn', f'tensor-meta patch skipped: {e}')
        return

    if not getattr(PerceptionTransformer, '_bevformer_onnx_tensor_meta_patch', False):
        def _get_bev_features_export(self, mlvl_feats, bev_queries, bev_h, bev_w,
                                     grid_length=[0.512, 0.512], bev_pos=None,
                                     prev_bev=None, **kwargs):
            bs = mlvl_feats[0].size(0)
            bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
            bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

            can_bus = kwargs['onnx_can_bus'].to(device=bev_queries.device,
                                                dtype=bev_queries.dtype)
            delta_x = can_bus[:, 0]
            delta_y = can_bus[:, 1]
            ego_angle = can_bus[:, -2] / torch.pi * 180.0
            grid_length_y = float(grid_length[0])
            grid_length_x = float(grid_length[1])
            translation_length = torch.sqrt(delta_x * delta_x + delta_y * delta_y)
            # ONNX lowers atan2 through division-like ops on this stack; protect
            # the no-motion case so first-frame can_bus zeros do not produce NaNs.
            eps = torch.ones_like(delta_x) * 1e-6
            safe_delta_x = torch.where(translation_length > eps, delta_x, torch.ones_like(delta_x))
            safe_delta_y = torch.where(translation_length > eps, delta_y, torch.zeros_like(delta_y))
            translation_angle = torch.atan2(safe_delta_y, safe_delta_x) / torch.pi * 180.0
            bev_angle = ego_angle - translation_angle
            shift_y = translation_length * torch.cos(bev_angle / 180.0 * torch.pi) / grid_length_y / float(bev_h)
            shift_x = translation_length * torch.sin(bev_angle / 180.0 * torch.pi) / grid_length_x / float(bev_w)
            shift = torch.stack((shift_x, shift_y), dim=1) * float(self.use_shift)

            if prev_bev is not None:
                if prev_bev.shape[1] == bev_h * bev_w:
                    prev_bev = prev_bev.permute(1, 0, 2)
                if self.rotate_prev_bev:
                    prev_bev = rotate_prev_bev_export(
                        prev_bev, can_bus[:, -1], bev_h, bev_w)

            can_bus_embed = self.can_bus_mlp(can_bus)[None, :, :]
            bev_queries = bev_queries + can_bus_embed * float(self.use_can_bus)

            feat_flatten = []
            spatial_shape_values = []
            for lvl, feat in enumerate(mlvl_feats):
                bs, num_cam, c, h, w = feat.shape
                spatial_shape_values.append((h, w))
                feat = feat.flatten(3).permute(1, 0, 3, 2)
                if self.use_cams_embeds:
                    feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
                feat = feat + self.level_embeds[None, None, lvl:lvl + 1, :].to(feat.dtype)
                feat_flatten.append(feat)

            feat_flatten = torch.cat(feat_flatten, 2)
            spatial_shapes = torch.as_tensor(
                spatial_shape_values, dtype=torch.long, device=bev_pos.device)
            level_start_index = torch.cat((spatial_shapes.new_zeros((1,)),
                                           spatial_shapes.prod(1).cumsum(0)[:-1]))
            feat_flatten = feat_flatten.permute(0, 2, 1, 3)

            self.encoder.onnx_lidar2img = kwargs['onnx_lidar2img']
            self.encoder.onnx_img_hw = kwargs['onnx_img_hw']
            bev_embed = self.encoder(
                bev_queries, feat_flatten, feat_flatten,
                bev_h=bev_h, bev_w=bev_w, bev_pos=bev_pos,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                prev_bev=prev_bev,
                shift=shift,
                **kwargs)
            return bev_embed

        PerceptionTransformer.get_bev_features = _get_bev_features_export
        PerceptionTransformer._bevformer_onnx_tensor_meta_patch = True
        report.add('export.patch', 'ok',
                   'patched PerceptionTransformer.get_bev_features to tensor meta path',
                   'Pass can_bus/lidar2img/img_hw as ONNX inputs; keep scene bookkeeping outside the graph.')

    try:
        import torch.nn.functional as _torch_F
        import mmcv.ops.multi_scale_deform_attn as _mmcv_msda_mod
        from projects.mmdet3d_plugin.bevformer.modules import multi_scale_deformable_attn_function as _msda_mod
        from projects.mmdet3d_plugin.bevformer.modules import decoder as _decoder_mod
        _msda_mod.F = _torch_F
        _mmcv_msda_mod.F = _torch_F
    except Exception as e:
        report.add('export.patch', 'warn', f'MSDA fallback patch skipped: {e}')
    else:
        if os.environ.get('BEVFORMER_KEEP_NATIVE_MSDA', '0') != '1':
            _msda_mod._can_use_ms_deform_attn_ext = lambda value: False
        report.add('export.patch', 'ok',
                   'forced _multi_scale_deform_attn to PyTorch fallback',
                   'MSDeformableAttention lowers through standard torch ops instead of MUSA ext.')

        CustomMSDeformableAttention = _decoder_mod.CustomMSDeformableAttention
        if (os.environ.get('BEVFORMER_KEEP_NATIVE_MSDA', '0') != '1'
                and not getattr(CustomMSDeformableAttention, '_bevformer_onnx_msda_patch', False)):
            def _custom_msda_forward_export(self, query, key=None, value=None,
                                            identity=None, query_pos=None,
                                            key_padding_mask=None,
                                            reference_points=None,
                                            spatial_shapes=None,
                                            level_start_index=None, **kwargs):
                assert self.batch_first is False and key_padding_mask is None
                if value is None:
                    value = query
                if identity is None:
                    identity = query
                if query_pos is not None:
                    query = query + query_pos
                query = query.permute(1, 0, 2)
                value = value.permute(1, 0, 2)
                bs, num_query, _ = query.shape
                _, num_value, _ = value.shape
                assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
                value = self.value_proj(value)
                value = value.view(bs, num_value, self.num_heads, -1)
                sampling_offsets = self.sampling_offsets(query).view(
                    bs, num_query, self.num_heads, self.num_levels,
                    self.num_points, 2)
                attention_weights = self.attention_weights(query).view(
                    bs, num_query, self.num_heads,
                    self.num_levels * self.num_points)
                attention_weights = attention_weights.softmax(-1)
                attention_weights = attention_weights.view(
                    bs, num_query, self.num_heads, self.num_levels,
                    self.num_points)
                if reference_points.shape[-1] == 2:
                    offset_normalizer = torch.stack(
                        [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
                    sampling_locations = reference_points[:, :, None, :, None, :] \
                        + sampling_offsets \
                        / offset_normalizer[None, None, None, :, None, :]
                elif reference_points.shape[-1] == 4:
                    sampling_locations = reference_points[:, :, None, :, None, :2] \
                        + sampling_offsets / self.num_points \
                        * reference_points[:, :, None, :, None, 2:] * 0.5
                else:
                    raise ValueError('reference_points last dim must be 2 or 4')
                output = _decoder_mod.multi_scale_deformable_attn_pytorch(
                    value, spatial_shapes, sampling_locations,
                    attention_weights)
                output = self.output_proj(output)
                output = output.permute(1, 0, 2)
                return self.dropout(output) + identity

            CustomMSDeformableAttention.forward = _custom_msda_forward_export
            CustomMSDeformableAttention._bevformer_onnx_msda_patch = True
            report.add('export.patch', 'ok',
                       'patched CustomMSDeformableAttention.forward to PyTorch fallback',
                       'Decoder deformable attention no longer calls MultiScaleDeformableAttnFunction on MUSA during export.')

    if (os.environ.get('BEVFORMER_KEEP_NATIVE_SCA', '0') != '1'
            and not getattr(SpatialCrossAttention, '_bevformer_onnx_dense_patch', False)):
        def _spatial_cross_attention_dense_export(
                self, query, key, value, residual=None, query_pos=None,
                key_padding_mask=None, reference_points=None,
                spatial_shapes=None, reference_points_cam=None, bev_mask=None,
                level_start_index=None, flag='encoder', **kwargs):
            """Export SCA without data-dependent Python indexing.

            Every camera processes all BEV queries. Invalid camera/query
            outputs are masked before aggregation. Attention is independent per
            query, so this is equivalent to compacting visible queries while
            avoiding a sample-specific traced max_len.
            """
            del key_padding_mask, reference_points, flag, kwargs
            if key is None:
                key = query
            if value is None:
                value = key

            inp_residual = query if residual is None else residual
            if query_pos is not None:
                query = query + query_pos

            bs, num_query, _ = query.shape
            depth = reference_points_cam.size(3)
            valid = bev_mask.any(dim=-1).permute(1, 0, 2)
            queries_all = query[:, None, :, :].expand(
                bs, self.num_cams, num_query, self.embed_dims)
            reference_points_all = reference_points_cam.permute(
                1, 0, 2, 3, 4)

            # Fixed-shape compact path. Stable scores place valid queries first
            # while preserving ascending query order within valid/invalid groups.
            query_id = torch.arange(
                num_query, device=query.device, dtype=torch.long)
            order_score = (valid.to(torch.long) * (2 * num_query)
                           - query_id[None, None, :])
            _, order = torch.sort(order_score, dim=2, descending=True)
            query_index = order[..., None].expand(
                bs, self.num_cams, num_query, self.embed_dims)
            ref_index = order[..., None, None].expand(
                bs, self.num_cams, num_query, depth, 2)
            queries_rebatch = torch.gather(queries_all, 2, query_index)
            reference_points_rebatch = torch.gather(
                reference_points_all, 2, ref_index)
            valid_sorted = torch.gather(valid, 2, order)
            valid_sorted_f = valid_sorted.to(dtype=queries_rebatch.dtype)
            queries_rebatch = queries_rebatch * valid_sorted_f[..., None]
            reference_points_rebatch = (
                reference_points_rebatch
                * valid_sorted_f[..., None, None])
            _, length, _, _ = key.shape
            key = key.permute(2, 0, 1, 3).reshape(
                bs * self.num_cams, length, self.embed_dims)
            value = value.permute(2, 0, 1, 3).reshape(
                bs * self.num_cams, length, self.embed_dims)
            attended = self.deformable_attention(
                query=queries_rebatch.reshape(
                    bs * self.num_cams, num_query, self.embed_dims),
                key=key,
                value=value,
                reference_points=reference_points_rebatch.reshape(
                    bs * self.num_cams, num_query, depth, 2),
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
            ).reshape(bs, self.num_cams, num_query, self.embed_dims)

            # Invert the compact permutation before camera aggregation.
            inverse_order = torch.argsort(order, dim=2)
            attended = torch.gather(
                attended, 2,
                inverse_order[..., None].expand(
                    bs, self.num_cams, num_query, self.embed_dims))
            valid_f = valid.to(dtype=attended.dtype)
            slots = (attended * valid_f[..., None]).sum(dim=1)
            count = valid_f.sum(dim=1).clamp(min=1.0)
            slots = self.output_proj(slots / count[..., None])
            return self.dropout(slots) + inp_residual

        SpatialCrossAttention.forward = _spatial_cross_attention_dense_export
        SpatialCrossAttention._bevformer_onnx_dense_patch = True
        report.add(
            'export.patch', 'ok',
            'patched SpatialCrossAttention to fixed all-query dense masking',
            'Avoid tracing a frame-specific Python max_len; runtime supports '
            'different camera visibility masks without graph shape changes.')

    if not getattr(BEVFormerEncoder, '_bevformer_onnx_tensor_meta_patch', False):
        def _point_sampling_export(self, reference_points, pc_range, img_metas):
            lidar2img = self.onnx_lidar2img.to(device=reference_points.device,
                                               dtype=torch.float32)
            img_hw = self.onnx_img_hw.to(device=reference_points.device,
                                         dtype=torch.float32)
            reference_points = reference_points.clone().to(torch.float32)
            reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
            reference_points[..., 1:2] = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
            reference_points[..., 2:3] = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
            reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
            reference_points = reference_points.permute(1, 0, 2, 3)
            d, b, num_query = reference_points.size()[:3]
            num_cam = lidar2img.size(1)
            reference_points = reference_points.view(d, b, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)
            lidar2img = lidar2img.view(1, b, num_cam, 1, 4, 4).repeat(d, 1, 1, num_query, 1, 1)
            reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)
            eps = 1e-5
            bev_mask = reference_points_cam[..., 2:3] > eps
            reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
                reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)
            reference_points_cam[..., 0] = reference_points_cam[..., 0] / img_hw[1]
            reference_points_cam[..., 1] = reference_points_cam[..., 1] / img_hw[0]
            bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                        & (reference_points_cam[..., 1:2] < 1.0)
                        & (reference_points_cam[..., 0:1] < 1.0)
                        & (reference_points_cam[..., 0:1] > 0.0))
            bev_mask = torch.nan_to_num(bev_mask)
            reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
            bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)
            return reference_points_cam, bev_mask

        BEVFormerEncoder.point_sampling = _point_sampling_export
        BEVFormerEncoder._bevformer_onnx_tensor_meta_patch = True
        report.add('export.patch', 'ok',
                   'patched BEVFormerEncoder.point_sampling to tensor lidar2img path',
                   'Do not rebuild lidar2img/img_shape through numpy/img_metas during export.')


# ---------------------------------------------------------------------------
# ONNX-friendly wrapper
# ---------------------------------------------------------------------------

class BEVFormerOnnxWrapper(torch.nn.Module):
    """Tensor-only I/O wrapper around BEVFormer for export probing.

    Inputs
      img:       [B, N, 3, H, W]
      prev_bev:  [B, bev_h*bev_w, C]  (zeros = no history)
      can_bus:   [B, 18]
      lidar2img: [B, N, 4, 4]

    Outputs (raw heads; no NMS)
      bev_embed:      [bev_h*bev_w, B, C] or [B, bev_h*bev_w, C]
      all_cls_scores: [num_dec, B, num_query, num_classes]
      all_bbox_preds: [num_dec, B, num_query, code_size]
    """

    def __init__(self, model: torch.nn.Module, img_h: int, img_w: int,
                 num_cams: int = 6):
        super().__init__()
        self.model = model
        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self.num_cams = int(num_cams)

        # Inference-friendly flags
        self.model.use_grid_mask = False
        self.model.video_test_mode = False
        if hasattr(self.model, 'prev_frame_info'):
            self.model.prev_frame_info = {
                'prev_bev': None,
                'scene_token': None,
                'prev_pos': 0,
                'prev_angle': 0,
            }

        # Unwrap torch.compile if present (must stay a bound method).
        from types import MethodType
        head = getattr(self.model, 'pts_bbox_head', None)
        if head is not None and hasattr(head, 'post_precess'):
            fn = head.post_precess
            orig = None
            if hasattr(fn, '_orig_mod'):
                orig = fn._orig_mod
            elif hasattr(fn, '__wrapped__'):
                orig = fn.__wrapped__
            if orig is not None:
                if isinstance(orig, torch.nn.Module):
                    head.post_precess = orig  # type: ignore[assignment]
                else:
                    head.post_precess = MethodType(orig, head)  # type: ignore[assignment]
            try:
                head.post_precess = torch.compiler.disable(head.post_precess)  # type: ignore[assignment]
            except Exception:
                pass

    def _build_img_metas(self, can_bus: torch.Tensor,
                         lidar2img: torch.Tensor) -> list:
        """Rebuild legacy img_metas from tensors (still Python; ONNX blocker)."""
        B = can_bus.shape[0]
        N = int(lidar2img.shape[1])
        img_shape = [(self.img_h, self.img_w, 3) for _ in range(N)]
        metas = []
        for b in range(B):
            # Keep lidar2img as (N,4,4) ndarray so encoder.point_sampling's
            # np.asarray([..]) yields (B,N,4,4). list(...) drops the batch dim
            # under tracer and triggers view errors.
            cb = can_bus[b].detach().float().cpu().numpy()
            l2i = lidar2img[b].detach().float().cpu().numpy()
            metas.append({
                'can_bus': cb,
                'lidar2img': l2i,
                'img_shape': img_shape,
                'pad_shape': img_shape,
                'prev_bev_exists': True,
                'scene_token': 'onnx_export',
            })
        return metas

    def forward(self, img: torch.Tensor, prev_bev: torch.Tensor,
                can_bus: torch.Tensor, lidar2img: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Avoid BEVFormer.extract_img_feat's `img.squeeze_()` when B==1, which
        # collapses [1,N,C,H,W] -> [N,C,H,W] and breaks traced batch dims.
        img_feats = self._extract_feat_no_squeeze(img)
        return self._head_forward_tensor_meta(img_feats, prev_bev, can_bus, lidar2img)

    def _head_forward_tensor_meta(self, mlvl_feats, prev_bev, can_bus, lidar2img):
        head = self.model.pts_bbox_head
        bs, _, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = head.query_embedding.weight.to(dtype)
        bev_queries = head.bev_embedding.weight.to(dtype)
        bev_mask = torch.zeros((bs, head.bev_h, head.bev_w),
                               device=bev_queries.device, dtype=dtype)
        bev_pos = head.positional_encoding(bev_mask).to(dtype)
        img_hw = torch.stack((can_bus.new_tensor(float(self.img_h)),
                              can_bus.new_tensor(float(self.img_w))))
        img_shape = [(self.img_h, self.img_w, 3) for _ in range(self.num_cams)]
        img_metas = [{
            'can_bus': [0.0] * 18,
            'lidar2img': [[[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]] for _ in range(self.num_cams)],
            'img_shape': img_shape,
            'pad_shape': img_shape,
            'prev_bev_exists': True,
            'scene_token': 'onnx_export',
        } for _ in range(bs)]
        outputs = head.transformer(
            mlvl_feats,
            bev_queries,
            object_query_embeds,
            head.bev_h,
            head.bev_w,
            grid_length=(head.real_h / head.bev_h, head.real_w / head.bev_w),
            bev_pos=bev_pos,
            reg_branches=head.reg_branches if head.with_box_refine else None,
            cls_branches=head.cls_branches if head.as_two_stage else None,
            img_metas=img_metas,
            prev_bev=prev_bev,
            onnx_can_bus=can_bus,
            onnx_lidar2img=lidar2img,
            onnx_img_hw=img_hw,
        )
        bev_embed, hs, init_reference, inter_references = outputs
        if hasattr(head, 'post_precess'):
            outputs_classes, outputs_coords = head.post_precess(
                hs, init_reference, inter_references)
        else:
            outputs_classes, outputs_coords = self._post_process_fallback(
                head, hs, init_reference, inter_references)
        return bev_embed, outputs_classes, outputs_coords

    @staticmethod
    def _post_process_fallback(head, hs, init_reference, inter_references):
        from mmdet.models.utils.transformer import inverse_sigmoid

        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            reference = (init_reference if lvl == 0
                         else inter_references[lvl - 1])
            reference = inverse_sigmoid(reference)
            outputs_class = head.cls_branches[lvl](hs[lvl])
            tmp = head.reg_branches[lvl](hs[lvl])
            if reference.shape[-1] != 3:
                raise RuntimeError('BEVFormer reference point dim must be 3')
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (tmp[..., 0:1] *
                             (head.pc_range[3] - head.pc_range[0]) +
                             head.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] *
                             (head.pc_range[4] - head.pc_range[1]) +
                             head.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] *
                             (head.pc_range[5] - head.pc_range[2]) +
                             head.pc_range[2])
            outputs_classes.append(outputs_class)
            outputs_coords.append(tmp)
        return torch.stack(outputs_classes), torch.stack(outputs_coords)

    def _extract_feat_no_squeeze(self, img: torch.Tensor):
        model = self.model
        target_device = next(model.img_backbone.parameters()).device
        target_dtype = next(model.img_backbone.parameters()).dtype
        img = img.to(target_device).to(target_dtype)
        B, N, C, H, W = img.shape
        x = img.reshape(B * N, C, H, W)
        if getattr(model, 'use_backbone_cl', False):
            x = x.to(memory_format=torch.channels_last)
        feats = model.img_backbone(x)
        if isinstance(feats, dict):
            feats = list(feats.values())
        if model.with_img_neck:
            feats = model.img_neck(feats)
        if getattr(model, 'use_backbone_cl', False):
            feats = [f.contiguous() for f in feats]
        out = []
        for f in feats:
            BN, Cc, Hh, Ww = f.shape
            out.append(f.view(B, N, Cc, Hh, Ww))
        return out

class BackboneNeckWrapper(torch.nn.Module):
    """Export-only: image backbone + FPN (no DCN-free guarantee)."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.backbone = model.img_backbone
        self.neck = model.img_neck if model.with_img_neck else None
        model.use_grid_mask = False

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # img: [B*N, 3, H, W]
        feats = self.backbone(img)
        if self.neck is not None:
            feats = self.neck(feats)
        return tuple(feats)


# ---------------------------------------------------------------------------
# Build / load
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Export BEVFormer to ONNX')
    p.add_argument('config', help='config file path')
    p.add_argument('checkpoint', nargs='?', default=None,
                   help='checkpoint (.pth); optional for structure-only probe')
    p.add_argument('--out', default='work_dirs/bevformer_onnx/bevformer.onnx')
    p.add_argument('--report', default=None,
                   help='markdown report path (default: <out>.report.md)')
    p.add_argument('--device', default='musa:0',
                   help='musa:0 / cuda:0 / cpu')
    p.add_argument('--opset', type=int, default=17)
    p.add_argument('--img-h', type=int, default=928,
                   help='padded image height (nuScenes 900 -> pad/32 = 928)')
    p.add_argument('--img-w', type=int, default=1600)
    p.add_argument('--num-cams', type=int, default=6)
    p.add_argument('--sample-data-root', default=None,
                   help='use a real val sample from this nuScenes root for tracing')
    p.add_argument('--sample-index', type=int, default=0,
                   help='val sample index when --sample-data-root is set')
    p.add_argument('--skip-full-export', action='store_true')
    p.add_argument('--skip-backbone-export', action='store_true')
    p.add_argument('--dynamo', action='store_true',
                   help='use torch.onnx.export(..., dynamo=True) if available')
    p.add_argument('--skip-onnx-normalize', action='store_true',
                   help='skip ORT compatibility casts/checker after export')
    p.add_argument('--cfg-options', nargs='+', action=None, default=None)
    return p.parse_args()


def resolve_device(spec: str) -> torch.device:
    if spec.startswith('musa'):
        import torch_musa  # noqa: F401
        if not torch.musa.is_available():
            raise RuntimeError('MUSA requested but not available')
        return torch.device(spec)
    if spec.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but not available')
        return torch.device(spec)
    return torch.device('cpu')


def build_model(cfg_path: str, checkpoint: Optional[str], device: torch.device,
                report: Report):
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmdet3d.models import build_model
    try:
        from bev_opt_setup import apply_bev_opt
    except Exception:
        apply_bev_opt = None

    cfg = Config.fromfile(cfg_path)
    # Force export-friendly bev_opt off even if config enables them.
    if (os.environ.get('BEVFORMER_KEEP_BEV_OPT', '0') != '1'
            and hasattr(cfg, 'bev_opt') and cfg.bev_opt):
        for k in list(cfg.bev_opt.keys()):
            if isinstance(cfg.bev_opt[k], bool):
                cfg.bev_opt[k] = False
        if apply_bev_opt is not None:
            apply_bev_opt(cfg)
        report.add('config', 'info',
                   'forced cfg.bev_opt=* = False for export probing',
                   'Keep bev_opt disabled (or provide ONNX equivalents) '
                   'when shipping an export config.')

    if os.environ.get('BEVFORMER_KEEP_BEV_OPT', '0') == '1' and apply_bev_opt is not None:
        apply_bev_opt(cfg)

    # Strip DCN backend kw that stock mmcv may not accept
    try:
        dcn = cfg.model.img_backbone.get('dcn', None)
        if (os.environ.get('BEVFORMER_KEEP_BEV_OPT', '0') != '1'
                and dcn and 'backend' in dcn):
            report.add('config', 'warn',
                       f"removed img_backbone.dcn.backend={dcn['backend']!r} "
                       '(stock mmcv DCNv2)',
                       'Export config should not pass fuse_c/tilelang backend; '
                       'use stock DCNv2 or plain Conv replacement.')
            dcn = dict(dcn)
            dcn.pop('backend', None)
            cfg.model.img_backbone.dcn = dcn
    except Exception as e:
        report.add('config', 'warn', f'dcn backend strip skipped: {e}')

    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    if cfg.get('plugin', False):
        import importlib
        if hasattr(cfg, 'plugin_dir'):
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir).split('/')
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
        else:
            _module_dir = os.path.dirname(cfg_path).split('/')
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
        importlib.import_module(_module_path)
        report.add('plugin', 'ok', f'imported plugin {_module_path}')

    patch_export_only_ops(report)

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    model.eval()

    if checkpoint:
        load_checkpoint(model, checkpoint, map_location='cpu')
        report.add('checkpoint', 'ok', f'loaded {checkpoint}')
    else:
        report.add('checkpoint', 'warn',
                   'no checkpoint given; random weights (structure probe only)')

    # Move to device; channels_last disabled via env
    model = model.to(device)
    report.add('device', 'ok', f'model on {device}')
    return cfg, model



def make_real_sample_inputs(cfg, device: torch.device, data_root: str,
                            sample_index: int, report: Report):
    from projects.mmdet3d_plugin.datasets import custom_build_dataset
    test_cfg = copy.deepcopy(cfg.data.test)
    test_cfg.data_root = data_root
    test_cfg.ann_file = os.path.join(data_root, 'nuscenes_infos_temporal_val.pkl')
    test_cfg.test_mode = True
    test_cfg.pop('samples_per_gpu', None)
    dataset = custom_build_dataset(test_cfg)
    item = dataset[int(sample_index)]
    img_dc = item['img'][0]
    meta_dc = item['img_metas'][0]
    img = img_dc.data if hasattr(img_dc, 'data') else img_dc
    meta = meta_dc.data if hasattr(meta_dc, 'data') else meta_dc
    if isinstance(meta, list):
        meta = meta[0]
    img = img.unsqueeze(0).to(device=device, dtype=torch.float32)
    can_bus = torch.as_tensor(meta['can_bus'], device=device,
                              dtype=torch.float32).unsqueeze(0)
    lidar2img = torch.as_tensor(meta['lidar2img'], device=device,
                                dtype=torch.float32).unsqueeze(0)
    bev_h = cfg.model.pts_bbox_head.bev_h
    bev_w = cfg.model.pts_bbox_head.bev_w
    embed = cfg.model.pts_bbox_head.in_channels
    prev_bev = torch.zeros(1, bev_h * bev_w, embed, device=device,
                           dtype=torch.float32)
    report.add('inputs.real_sample', 'ok',
               f'idx={sample_index} img{tuple(img.shape)} can_bus{tuple(can_bus.shape)} '
               f'lidar2img{tuple(lidar2img.shape)} sample_idx={meta.get("sample_idx")}')
    return img, prev_bev, can_bus, lidar2img, bev_h, bev_w, embed

def make_dummy_inputs(cfg, device: torch.device, img_h: int, img_w: int,
                      num_cams: int):
    bev_h = cfg.model.pts_bbox_head.bev_h
    bev_w = cfg.model.pts_bbox_head.bev_w
    embed = cfg.model.pts_bbox_head.in_channels
    B, N = 1, num_cams
    img = torch.randn(B, N, 3, img_h, img_w, device=device, dtype=torch.float32)
    prev_bev = torch.zeros(B, bev_h * bev_w, embed, device=device,
                           dtype=torch.float32)
    can_bus = torch.zeros(B, 18, device=device, dtype=torch.float32)
    # Identity-ish extrinsics (valid enough for shape path)
    lidar2img = torch.eye(4, device=device, dtype=torch.float32).view(
        1, 1, 4, 4).repeat(B, N, 1, 1)
    return img, prev_bev, can_bus, lidar2img, bev_h, bev_w, embed


def try_call(report: Report, stage: str, fn: Callable, adapt_hint: str = '') -> Any:
    try:
        out = fn()
        report.add(stage, 'ok', 'succeeded')
        return out
    except Exception as e:
        tb = traceback.format_exc(limit=8)
        report.add(stage, 'fail', f'{type(e).__name__}: {e}',
                   adapt_hint or f'Investigate failure at stage `{stage}`.')
        print(tb)
        return None


def try_onnx_export(report: Report, stage: str, module: torch.nn.Module,
                    args: Sequence[torch.Tensor], out_path: str,
                    input_names: List[str], output_names: List[str],
                    opset: int, dynamo: bool, adapt_hint: str) -> bool:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    kwargs = dict(
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        do_constant_folding=True,
        verbose=False,
    )

    def _export():
        module.eval()
        with torch.no_grad():
            # Default to legacy exporter; dynamo path often trips on custom ops.
            export_kwargs = dict(kwargs)
            if dynamo:
                try:
                    torch.onnx.export(module, tuple(args), out_path,
                                      dynamo=True, **export_kwargs)
                    return
                except TypeError:
                    pass
            try:
                torch.onnx.export(module, tuple(args), out_path,
                                  dynamo=False, **export_kwargs)
            except TypeError:
                torch.onnx.export(module, tuple(args), out_path, **export_kwargs)

    try:
        _export()
        report.add(stage, 'ok', f'wrote {out_path}')
        return True
    except Exception as e:
        err = f'{type(e).__name__}: {e}'
        # Attach a short traceback tip for the report.
        tip = adapt_hint
        el = err.lower()
        if 'deform' in el or 'at() was not in the map' in el:
            tip = (adapt_hint + ' | runtime: DCNv2/mmcv custom op lacks ONNX '
                   f'symbolic ({err})')
        elif 'aten::mul' in el and 'bool' in el:
            tip = (adapt_hint + ' | runtime: Python bool×Tensor in '
                   f'get_bev_features use_shift/use_can_bus ({err.splitlines()[0]})')
        elif 'numpy' in el or ('tensor' in el and 'python' in el):
            tip = adapt_hint + f' | runtime: Python/numpy blocker ({err})'
        elif 'shape' in el and 'invalid' in el:
            tip = (adapt_hint + ' | runtime: traced batch/meta shape mismatch '
                   f'({err}); check squeeze_/lidar2img/point_sampling.')
        report.add(stage, 'fail', err.splitlines()[0], tip)
        print(traceback.format_exc(limit=12))
        return False


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as fp:
        for chunk in iter(lambda: fp.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_and_check_onnx(path: str, config: str,
                             checkpoint: Optional[str], report: Report) -> bool:
    """Normalize exporter edge cases required by ONNX Runtime CUDA."""
    try:
        import onnx
        from onnx import TensorProto, helper, shape_inference

        model = onnx.load(path)
        bool_values = set()
        for value in list(model.graph.input) + list(model.graph.value_info):
            tensor_type = value.type.tensor_type
            if tensor_type.elem_type == TensorProto.BOOL:
                bool_values.add(value.name)
        for initializer in model.graph.initializer:
            if initializer.data_type == TensorProto.BOOL:
                bool_values.add(initializer.name)

        bool_ops = {
            'And', 'Equal', 'Greater', 'GreaterOrEqual', 'IsNaN', 'Less',
            'LessOrEqual', 'Not', 'Or', 'Xor',
        }
        for node in model.graph.node:
            if node.op_type in bool_ops:
                bool_values.update(node.output)
            elif node.op_type == 'Cast':
                to_type = next(
                    (attr.i for attr in node.attribute if attr.name == 'to'),
                    None)
                if to_type == TensorProto.BOOL:
                    bool_values.update(node.output)

        new_nodes = []
        grid_casts = 0
        reduce_casts = 0
        used_names = {
            name for node in model.graph.node
            for name in list(node.input) + list(node.output)
        }

        def unique_name(base: str) -> str:
            name = base
            suffix = 0
            while name in used_names:
                suffix += 1
                name = f'{base}_{suffix}'
            used_names.add(name)
            return name

        for index, node in enumerate(model.graph.node):
            if node.op_type == 'GridSample' and len(node.input) > 1:
                cast_out = unique_name(f'onnx_grid_float_{index}')
                new_nodes.append(helper.make_node(
                    'Cast', [node.input[1]], [cast_out],
                    name=unique_name(f'CastGridToFloat_{index}'),
                    to=TensorProto.FLOAT))
                node.input[1] = cast_out
                grid_casts += 1
            if (node.op_type == 'ReduceSum' and node.input
                    and node.input[0] in bool_values):
                cast_out = unique_name(f'onnx_bool_int64_{index}')
                new_nodes.append(helper.make_node(
                    'Cast', [node.input[0]], [cast_out],
                    name=unique_name(f'CastBoolToInt64_{index}'),
                    to=TensorProto.INT64))
                node.input[0] = cast_out
                reduce_casts += 1
            new_nodes.append(node)

        model.graph.ClearField('node')
        model.graph.node.extend(new_nodes)
        props = {entry.key: entry.value for entry in model.metadata_props}
        props.update({
            'bevformer.source_config': os.path.abspath(config),
            'bevformer.export_sca': 'fixed-all-query-dense-mask',
            'bevformer.export_dcn': 'grid-sample-equivalent',
        })
        if checkpoint:
            props['bevformer.source_checkpoint'] = os.path.abspath(checkpoint)
            props['bevformer.source_checkpoint_sha256'] = _sha256(checkpoint)
        helper.set_model_props(model, props)
        onnx.checker.check_model(model, full_check=True)
        shape_inference.infer_shapes(model, strict_mode=True)
        onnx.save(model, path)
        report.add(
            'onnx.normalize', 'ok',
            f'checker/strict-shape passed; GridSample casts={grid_casts}, '
            f'bool ReduceSum casts={reduce_casts}')
        return True
    except Exception as e:
        report.add(
            'onnx.normalize', 'fail', f'{type(e).__name__}: {e}',
            'Install a matching onnx package and inspect the exported graph.')
        print(traceback.format_exc(limit=12))
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    report = Report()
    for title, adapt in STATIC_ADAPTATIONS:
        report.add('static', 'info', title, adapt)

    report_path = args.report or (args.out + '.report.md')
    device = resolve_device(args.device)

    cfg, model = build_model(args.config, args.checkpoint, device, report)
    if args.sample_data_root:
        img, prev_bev, can_bus, lidar2img, bev_h, bev_w, embed = make_real_sample_inputs(
            cfg, device, args.sample_data_root, args.sample_index, report)
    else:
        img, prev_bev, can_bus, lidar2img, bev_h, bev_w, embed = make_dummy_inputs(
            cfg, device, args.img_h, args.img_w, args.num_cams)
    report.add('inputs', 'info',
               f'img{tuple(img.shape)} prev_bev{tuple(prev_bev.shape)} '
               f'can_bus{tuple(can_bus.shape)} lidar2img{tuple(lidar2img.shape)}')

    wrapper = BEVFormerOnnxWrapper(model, args.img_h, args.img_w, args.num_cams)
    wrapper.eval()

    # ---- Stage: backbone(+neck) eager ----
    def _backbone_eager():
        bn = BackboneNeckWrapper(model)
        x = img.reshape(-1, 3, args.img_h, args.img_w)
        outs = bn(x)
        return tuple(o.shape for o in outs)

    shapes = try_call(
        report, 'eager.backbone_neck', _backbone_eager,
        adapt_hint='DCNv2 / MUSA conv path must run in eager before any ONNX export.')
    if shapes:
        report.add('eager.backbone_neck', 'info', f'feat shapes={shapes}')

    # ---- Stage: full wrapper eager ----
    def _full_eager():
        with torch.no_grad():
            bev, cls, box = wrapper(img, prev_bev, can_bus, lidar2img)
        return bev.shape, cls.shape, box.shape

    full_shapes = try_call(
        report, 'eager.full_wrapper', _full_eager,
        adapt_hint='Fix Python/numpy/custom-op issues in head/encoder until eager '
                   'wrapper forward succeeds, then retry ONNX.')
    if full_shapes:
        report.add('eager.full_wrapper', 'info',
                   f'bev/cls/box shapes={full_shapes}')

    # ---- Stage: backbone ONNX ----
    if not args.skip_backbone_export:
        bn = BackboneNeckWrapper(model)
        x = img.reshape(-1, 3, args.img_h, args.img_w)
        bb_out = os.path.splitext(args.out)[0] + '_backbone_neck.onnx'
        # Infer output count
        with torch.no_grad():
            n_outs = len(bn(x))
        backbone_ok = try_onnx_export(
            report, 'onnx.backbone_neck', bn, (x,), bb_out,
            input_names=['img_bn'],
            output_names=[f'feat_{i}' for i in range(n_outs)],
            opset=args.opset, dynamo=args.dynamo,
            adapt_hint='Backbone export blocked by DCNv2 (and/or MUSA-only '
                       'kernels). Options: (1) ONNX/ORT plugin for DCNv2, '
                       '(2) replace DCN stages with Conv2d for an export-only '
                       'config, (3) export on a build where mmcv registers '
                       'DCN symbolic.')
        if backbone_ok and not args.skip_onnx_normalize:
            normalize_and_check_onnx(
                bb_out, args.config, args.checkpoint, report)

    # ---- Stage: full ONNX ----
    if not args.skip_full_export:
        full_ok = try_onnx_export(
            report, 'onnx.full_wrapper', wrapper,
            (img, prev_bev, can_bus, lidar2img), args.out,
            input_names=['img', 'prev_bev', 'can_bus', 'lidar2img'],
            output_names=['bev_embed', 'all_cls_scores', 'all_bbox_preds'],
            opset=args.opset, dynamo=args.dynamo,
            adapt_hint='Full-graph export needs: tensorized metas, torch-only '
                       'shift/can_bus/point_sampling, MSDA pytorch/grid_sample '
                       'or plugin, DCN plugin/replacement, no host postprocess.')
        if full_ok and not args.skip_onnx_normalize:
            normalize_and_check_onnx(
                args.out, args.config, args.checkpoint, report)

    # Extra static notes from model structure
    if getattr(model, 'use_grid_mask', False):
        report.add('model', 'warn', 'use_grid_mask still True',
                   'Force use_grid_mask=False for export/inference graph.')
    dcn = cfg.model.img_backbone.get('dcn', None)
    if dcn:
        report.add('model', 'info',
                   f'backbone still configured with DCN: {dcn}',
                   'Decide DCN strategy before production ONNX.')

    text = report.dump(report_path)
    print('\n======== adaptation checklist (deduped) ========')
    in_list = False
    for line in text.splitlines():
        if line.startswith('## Deduped'):
            in_list = True
            continue
        if in_list and line.strip():
            print(line)


if __name__ == '__main__':
    main()
