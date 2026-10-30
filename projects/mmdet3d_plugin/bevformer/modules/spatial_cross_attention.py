
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import build_attention
import math
from mmcv.runner import force_fp32, auto_fp16

from mmcv.runner.base_module import BaseModule, ModuleList, Sequential

from mmcv.utils import ext_loader
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, \
    MultiScaleDeformableAttnFunction_fp16
from projects.mmdet3d_plugin.models.utils.bricks import run_time
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])

from typing import Optional, no_type_check
from .multi_scale_deformable_attn_function import _sca_python_scatter, _SCAAttentionScatterFunction, _SCAQueryReferenceGatherFunction, \
    _get_sca_mask_metadata, _multi_scale_deform_attn, _SCAAttentionScatterDivFunction, _SCAMSDeformAttnScatterDivFunction


import torch

@ATTENTION.register_module()
class SpatialCrossAttention(BaseModule):
    """An attention module used in BEVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4),
                 **kwargs
                 ):
        super(SpatialCrossAttention, self).__init__(init_cfg)

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    # @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    # def forward(self,
    #             query,
    #             key,
    #             value,
    #             residual=None,
    #             query_pos=None,
    #             key_padding_mask=None,
    #             reference_points=None,
    #             spatial_shapes=None,
    #             reference_points_cam=None,
    #             bev_mask=None,
    #             level_start_index=None,
    #             flag='encoder',
    #             **kwargs):
    #     """
    #     Optimized Forward Function of Detr3DCrossAtten.
    #     Removes nonzero() and loops for significant speedup.
    #     """
    #     if key is None: key = query
    #     if value is None: value = key

    #     if residual is None:
    #         inp_residual = query
    #         slots = torch.zeros_like(query)

    #     if query_pos is not None:
    #         query = query + query_pos

    #     bs, num_query, _ = query.size()
    #     num_cams = bev_mask.shape[0]

    #     # ----------------------------------------------------------------------
    #     # 1. 预计算索引与Mask (替代 nonzero + Loop)
    #     # ----------------------------------------------------------------------
    #     # 原始逻辑取 batch[0] 计算 mask。假设 sum(-1) > 0 代表投影有效。
    #     # mask_flatten: [num_cams, num_queries] (int 0 or 1)
    #     mask_flatten = (bev_mask[:, 0].sum(dim=-1) > 0).int()

    #     valid_lens = mask_flatten.sum(dim=1)
    #     # 【关键点】唯一的同步点，获取最大长度。比循环中的多次同步快得多。
    #     max_len = valid_lens.max().item()

    #     # 使用 sort 将有效索引 (1) 排在前面
    #     # sorted_indices: [num_cams, num_queries]
    #     _, sorted_indices = torch.sort(mask_flatten, dim=1, descending=True)

    #     # 截取前 max_len 个索引作为提取坐标
    #     # top_indices: [num_cams, max_len]
    #     top_indices = sorted_indices[:, :max_len]

    #     # 生成有效性 mask (用于后续过滤掉填充部分的脏数据)
    #     # valid_mask: [num_cams, max_len]
    #     seq_range = torch.arange(max_len, device=query.device).unsqueeze(0)
    #     valid_mask = seq_range < valid_lens.unsqueeze(1)

    #     # ----------------------------------------------------------------------
    #     # 2. 向量化 Gather (替代双重循环赋值)
    #     # ----------------------------------------------------------------------
    #     # 目标: 构造 queries_rebatch [bs, num_cams, max_len, dim]

    #     # 2.1 准备 Query
    #     # 扩展 query: [bs, 1, num_queries, dim] -> [bs, num_cams, num_queries, dim]
    #     query_expanded = query.unsqueeze(1).expand(-1, num_cams, -1, -1)

    #     # 扩展 indices 以匹配 gather 维度: [bs, num_cams, max_len, dim]
    #     gather_idx_query = top_indices.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, self.embed_dims)

    #     # 执行 Gather (在 num_queries 维度上抓取)
    #     queries_rebatch = torch.gather(query_expanded, 2, gather_idx_query)

    #     # 【重要】将 Padding 位置的数据置零，防止脏数据污染
    #     queries_rebatch = queries_rebatch * valid_mask.unsqueeze(0).unsqueeze(-1)

    #     # 2.2 准备 Reference Points
    #     # reference_points_cam: [num_cams, bs, num_queries, D, 2]
    #     # -> permute to [bs, num_cams, num_queries, D, 2]
    #     ref_pts_expanded = reference_points_cam.permute(1, 0, 2, 3, 4)
    #     D = ref_pts_expanded.shape[3]

    #     # 扩展 indices: [bs, num_cams, max_len, D, 2]
    #     gather_idx_ref = top_indices.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(bs, -1, -1, D, 2)

    #     # 执行 Gather
    #     reference_points_rebatch = torch.gather(ref_pts_expanded, 2, gather_idx_ref)

    #     # Mask 掉无效点
    #     reference_points_rebatch = reference_points_rebatch * valid_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    #     # ----------------------------------------------------------------------
    #     # 3. Attention Calculation (保持不变)
    #     # ----------------------------------------------------------------------
    #     num_cams, l, bs, embed_dims = key.shape
    #     key = key.permute(2, 0, 1, 3).reshape(bs * self.num_cams, l, self.embed_dims)
    #     value = value.permute(2, 0, 1, 3).reshape(bs * self.num_cams, l, self.embed_dims)

    #     queries = self.deformable_attention(
    #         query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims),
    #         key=key,
    #         value=value,
    #         reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, D, 2),
    #         spatial_shapes=spatial_shapes,
    #         level_start_index=level_start_index
    #     ).view(bs, self.num_cams, max_len, self.embed_dims)

    #     # ----------------------------------------------------------------------
    #     # 4. 向量化 Scatter (替代 slots += ... 循环)
    #     # ----------------------------------------------------------------------
    #     # 我们需要将计算好的 queries 加回 slots 的对应位置

    #     # 再次 Mask 掉 Attention 输出中的无效部分
    #     # 因为 top_indices 在 padding 处指向的是无效索引，如果不 mask，会把垃圾值加到那些索引上
    #     queries = queries * valid_mask.unsqueeze(0).unsqueeze(-1)

    #     # 展平以便 scatter
    #     # src_flat: [bs, num_cams * max_len, dim]
    #     src_flat = queries.reshape(bs, num_cams * max_len, self.embed_dims)

    #     # index_flat: [bs, num_cams * max_len, dim]
    #     # top_indices [num_cams, max_len] -> 展开并广播到 batch 和 dim
    #     idx_flat = top_indices.reshape(1, num_cams * max_len, 1).expand(bs, -1, self.embed_dims)

    #     # 使用 scatter_add_ 原地相加
    #     # slots shape: [bs, num_queries, dim]
    #     slots.scatter_add_(1, idx_flat, src_flat)

    #     # ----------------------------------------------------------------------
    #     # 5. 归一化
    #     # ----------------------------------------------------------------------
    #     count = bev_mask.sum(-1) > 0
    #     count = count.permute(1, 2, 0).sum(-1)
    #     count = torch.clamp(count, min=1.0)
    #     slots = slots / count[..., None]
    #     slots = self.output_proj(slots)

    #     return self.dropout(slots) + inp_residual

    # @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    # def forward(self,
    #             query,
    #             key,
    #             value,
    #             residual=None,
    #             query_pos=None,
    #             key_padding_mask=None,
    #             reference_points=None,
    #             spatial_shapes=None,
    #             reference_points_cam=None,
    #             bev_mask=None,
    #             level_start_index=None,
    #             flag='encoder',
    #             **kwargs):
    #     """Forward Function of Detr3DCrossAtten.
    #     Args:
    #         query (Tensor): Query of Transformer with shape
    #             (num_query, bs, embed_dims).
    #         key (Tensor): The key tensor with shape
    #             `(num_key, bs, embed_dims)`.
    #         value (Tensor): The value tensor with shape
    #             `(num_key, bs, embed_dims)`. (B, N, C, H, W)
    #         residual (Tensor): The tensor used for addition, with the
    #             same shape as `x`. Default None. If None, `x` will be used.
    #         query_pos (Tensor): The positional encoding for `query`.
    #             Default: None.
    #         key_pos (Tensor): The positional encoding for  `key`. Default
    #             None.
    #         reference_points (Tensor):  The normalized reference
    #             points with shape (bs, num_query, 4),
    #             all elements is range in [0, 1], top-left (0,0),
    #             bottom-right (1, 1), including padding area.
    #             or (N, Length_{query}, num_levels, 4), add
    #             additional two dimensions is (w, h) to
    #             form reference boxes.
    #         key_padding_mask (Tensor): ByteTensor for `query`, with
    #             shape [bs, num_key].
    #         spatial_shapes (Tensor): Spatial shape of features in
    #             different level. With shape  (num_levels, 2),
    #             last dimension represent (h, w).
    #         level_start_index (Tensor): The start index of each level.
    #             A tensor has shape (num_levels) and can be represented
    #             as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
    #     Returns:
    #          Tensor: forwarded results with shape [num_query, bs, embed_dims].
    #     """

    #     if key is None:
    #         key = query
    #     if value is None:
    #         value = key

    #     if residual is None:
    #         inp_residual = query
    #         slots = torch.zeros_like(query)
    #     if query_pos is not None:
    #         query = query + query_pos

    #     bs, num_query, _ = query.size()

    #     D = reference_points_cam.size(3)
    #     indexes = []
    #     for i, mask_per_img in enumerate(bev_mask):
    #         index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
    #         indexes.append(index_query_per_img)
    #     max_len = max([len(each) for each in indexes])

    #     # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
    #     queries_rebatch = query.new_zeros(
    #         [bs, self.num_cams, max_len, self.embed_dims])
    #     reference_points_rebatch = reference_points_cam.new_zeros(
    #         [bs, self.num_cams, max_len, D, 2])

    #     for j in range(bs):
    #         for i, reference_points_per_img in enumerate(reference_points_cam):
    #             index_query_per_img = indexes[i]
    #             queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
    #             reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]

    #     num_cams, l, bs, embed_dims = key.shape

    #     key = key.permute(2, 0, 1, 3).reshape(
    #         bs * self.num_cams, l, self.embed_dims)
    #     value = value.permute(2, 0, 1, 3).reshape(
    #         bs * self.num_cams, l, self.embed_dims)

    #     queries = self.deformable_attention(query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims), key=key, value=value,
    #                                         reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, D, 2), spatial_shapes=spatial_shapes,
    #                                         level_start_index=level_start_index).view(bs, self.num_cams, max_len, self.embed_dims)
    #     for j in range(bs):
    #         for i, index_query_per_img in enumerate(indexes):
    #             # slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]
    #             n = index_query_per_img.numel()
    #             if n > 0:
    #                 idx = index_query_per_img.contiguous().long()
    #                 src = queries[j, i, :n].contiguous()
    #                 slots[j].index_add_(0, idx, src)

    #     count = bev_mask.sum(-1) > 0
    #     count = count.permute(1, 2, 0).sum(-1)
    #     count = torch.clamp(count, min=1.0)
    #     slots = slots / count[..., None]
    #     slots = self.output_proj(slots)

    #     return self.dropout(slots) + inp_residual

    @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def forward(self,
                query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                residual: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                reference_points: Optional[torch.Tensor] = None,
                spatial_shapes: Optional[torch.Tensor] = None,
                reference_points_cam: Optional[torch.Tensor] = None,
                bev_mask: Optional[torch.Tensor] = None,
                level_start_index: Optional[torch.Tensor] = None,
                flag: str = 'encoder',
                **kwargs) -> torch.Tensor:
        del reference_points, key_padding_mask, flag
        if key is None:
            key = query
        if value is None:
            value = key

        if not self.batch_first:
            query = query.permute(1, 0, 2)
            if residual is not None:
                residual = residual.permute(1, 0, 2)
            if query_pos is not None:
                query_pos = query_pos.permute(1, 0, 2)

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        else:
            inp_residual = residual
            slots = torch.zeros_like(residual)
        if query_pos is not None:
            query = query + query_pos

        bs, _, _ = query.size()
        depth = reference_points_cam.size(3)
        sca_mask_metadata = kwargs.get('sca_mask_metadata')
        if sca_mask_metadata is None:
            sca_mask_metadata = _get_sca_mask_metadata(bev_mask)
        indexes, valid_lens, max_len, count = sca_mask_metadata
        if max_len == 0:
            output = self.dropout(self.output_proj(slots)) + inp_residual
            if not self.batch_first:
                output = output.permute(1, 0, 2)
            return output

        use_fused_gather_kernel = (
            torch.is_grad_enabled()
            and (query.requires_grad or reference_points_cam.requires_grad))
        if use_fused_gather_kernel:
            queries_rebatch, reference_points_rebatch = \
                _SCAQueryReferenceGatherFunction.apply(
                    query, reference_points_cam, indexes, self.num_cams,
                    max_len)
        else:
            queries_rebatch = query.new_zeros(
                (bs, self.num_cams, max_len, self.embed_dims))
            reference_points_rebatch = reference_points_cam.new_zeros(
                (bs, self.num_cams, max_len, depth, 2))
        if not use_fused_gather_kernel:
            for cam_id, reference_points_per_img in enumerate(
                    reference_points_cam):
                index_query_per_img = indexes[cam_id]
                valid_len = valid_lens[cam_id]
                if valid_len == 0:
                    continue
                queries_rebatch[:, cam_id, :valid_len] = torch.index_select(
                    query, 1, index_query_per_img)
                reference_points_rebatch[:, cam_id, :valid_len] = \
                    torch.index_select(reference_points_per_img, 1,
                                       index_query_per_img)

        num_cams, num_value, bs, _ = value.shape
        value = value.permute(2, 0, 1, 3).reshape(bs * num_cams, num_value,
                                                  self.embed_dims)

        use_direct_scatter = (
            torch.musa.is_available() and value.device.type == 'musa'
            and self.deformable_attention.batch_first
            and self.deformable_attention.embed_dims == self.embed_dims
            and self.embed_dims // self.deformable_attention.num_heads == 32
            and reference_points_rebatch.shape[-1] == 2)
        if use_direct_scatter:
            slots = self.deformable_attention.forward_scatter_div(
                query=queries_rebatch.view(bs * self.num_cams, max_len,
                                           self.embed_dims),
                value=value,
                reference_points=reference_points_rebatch.view(
                    bs * self.num_cams, max_len, depth, 2),
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                indexes=indexes,
                count=count,
                num_query=query.size(1),
                num_cams=self.num_cams,
                max_len=max_len)
        else:
            queries = self.deformable_attention(
                query=queries_rebatch.view(bs * self.num_cams, max_len,
                                           self.embed_dims),
                key=None,
                value=value,
                reference_points=reference_points_rebatch.view(
                    bs * self.num_cams, max_len, depth, 2),
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index).view(bs, self.num_cams,
                                                          max_len,
                                                          self.embed_dims)
            if torch.is_grad_enabled() and queries.requires_grad:
                slots = _SCAAttentionScatterDivFunction.apply(
                    queries, indexes, count, query.size(1))
            else:
                slots = _sca_python_scatter(queries, indexes, valid_lens,
                                            query.size(1))
                count = count.to(slots.dtype)
                slots = slots / count[..., None]
        output = self.dropout(self.output_proj(slots)) + inp_residual
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        return output


    @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def forward_bk(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                bev_mask=None,
                level_start_index=None,
                flag='encoder',
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for  `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        # from .dump import dump_forward_inputs_once
        # dump_forward_inputs_once(
        #     tag="spatial_cross_attn_forward",
        #     inputs_dict=dict(
        #         query=query,
        #         key=key,
        #         value=value,
        #         residual=residual,
        #         query_pos=query_pos,
        #         key_padding_mask=key_padding_mask,
        #         reference_points=reference_points,
        #         spatial_shapes=spatial_shapes,
        #         reference_points_cam=reference_points_cam,
        #         bev_mask=bev_mask,
        #         level_start_index=level_start_index,
        #         flag=flag,
        #         kwargs=kwargs,
        #     ),
        #     save_dir="/tmp/bevformer_forward_dump",
        #     cpu=True,
        # )
        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.size()

        D = reference_points_cam.size(3)
        # indexes = []
        # for i, mask_per_img in enumerate(bev_mask):
        #     index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
        #     indexes.append(index_query_per_img)
        # max_len = max([len(each) for each in indexes])

        # # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
        # queries_rebatch = query.new_zeros(
        #     [bs, self.num_cams, max_len, self.embed_dims])
        # reference_points_rebatch = reference_points_cam.new_zeros(
        #     [bs, self.num_cams, max_len, D, 2])

        # for j in range(bs):
        #     for i, reference_points_per_img in enumerate(reference_points_cam):
        #         index_query_per_img = indexes[i]
        #         queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
        #         reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]

        # ---------------------------------------------------------
        # 1. 生成每个 camera 对应的有效 query index
        # ---------------------------------------------------------
        # reference_points_cam: [num_cams, bs, num_query, D, 2]
        # bev_mask: usually [num_cams, bs, num_query, D]
        #
        # 原始逻辑是：
        #   for i, mask_per_img in enumerate(bev_mask):
        #       index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
        #
        # 所以这里保持只使用 batch 0
        valid = bev_mask[:, 0].sum(-1) > 0       # [num_cams, num_query]

        valid_counts = valid.sum(-1)             # [num_cams]
        # max_len = int(valid_counts.max().item())
        max_len = 10000

        # 把有效 query 排到前面
        # valid=True -> ~valid=False -> 0，排在前面
        idx_sorted = torch.argsort((~valid).to(torch.int32), dim=-1)

        # [num_cams, max_len]
        idx_padded = idx_sorted[:, :max_len]

        # [num_cams, max_len]
        valid_padded = valid.gather(1, idx_padded)

        # ---------------------------------------------------------
        # 2. gather query -> queries_rebatch
        # ---------------------------------------------------------
        # query: [bs, num_query, C]
        # queries_rebatch: [bs, num_cams, max_len, C]
        query_expand = query[:, None, :, :].expand(
            bs, self.num_cams, num_query, self.embed_dims
        )

        idx_query = idx_padded[None, :, :, None].expand(
            bs, self.num_cams, max_len, self.embed_dims
        )

        queries_rebatch = torch.gather(
            query_expand,
            dim=2,
            index=idx_query,
        )

        # padding 位置清 0，等价于原来的 new_zeros + 部分赋值
        queries_rebatch = queries_rebatch * valid_padded[None, :, :, None].to(queries_rebatch.dtype)

        # ---------------------------------------------------------
        # 3. gather reference_points_cam -> reference_points_rebatch
        # ---------------------------------------------------------
        # reference_points_cam: [num_cams, bs, num_query, D, 2]
        # 转成 [bs, num_cams, num_query, D, 2]
        reference_points_cam_batched = reference_points_cam.permute(1, 0, 2, 3, 4)

        idx_ref = idx_padded[None, :, :, None, None].expand(
            bs, self.num_cams, max_len, D, 2
        )

        reference_points_rebatch = torch.gather(
            reference_points_cam_batched,
            dim=2,
            index=idx_ref,
        )

        reference_points_rebatch = reference_points_rebatch * valid_padded[None, :, :, None, None].to(
            reference_points_rebatch.dtype
        )

        num_cams, l, bs, embed_dims = key.shape

        key = key.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)

        queries = self.deformable_attention(query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims), key=key, value=value,
                                            reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, D, 2), spatial_shapes=spatial_shapes,
                                            level_start_index=level_start_index).view(bs, self.num_cams, max_len, self.embed_dims)
        # for j in range(bs):
        #     for i, index_query_per_img in enumerate(indexes):
        #         slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]
        # padding 位置清 0，避免 scatter_add 写回无效位置
        # queries = queries * valid_padded[None, :, :, None].to(queries.dtype)

        # # ---------------------------------------------------------
        # # 6. scatter_add 写回 slots
        # # ---------------------------------------------------------
        # # queries: [bs, num_cams, max_len, C]
        # # slots:   [bs, num_query, C]
        # queries_flat = queries.reshape(bs, self.num_cams * max_len, self.embed_dims)

        # idx_flat = idx_padded.reshape(1, self.num_cams * max_len, 1).expand(
        #     bs, self.num_cams * max_len, self.embed_dims
        # )

        # slots.scatter_add_(
        #     dim=1,
        #     index=idx_flat,
        #     src=queries_flat,
        # )
        slots = scatter_slots(
            queries,
            idx_padded,
            valid_padded,
            bs,
            self.num_cams,
            max_len,
            self.embed_dims,
            num_query,
        )

        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual

def scatter_slots(
    queries,
    idx_padded,
    valid_padded,
    bs: int,
    num_cams: int,
    max_len: int,
    embed_dims: int,
    num_query: int,
):
    queries = queries * valid_padded[None, :, :, None].to(queries.dtype)

    queries_flat = queries.reshape(
        bs, num_cams * max_len, embed_dims
    )

    idx_flat = idx_padded.reshape(1, num_cams * max_len, 1).expand(
        bs, num_cams * max_len, embed_dims
    )

    slots = queries.new_zeros(bs, num_query, embed_dims)

    slots.scatter_add_(
        dim=1,
        index=idx_flat,
        src=queries_flat,
    )

    return slots


@ATTENTION.register_module()
class MSDeformableAttention3D(BaseModule):
    """An attention module used in BEVFormer based on Deformable-Detr.
    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=8,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward_scatter_div(self,
                            query: torch.Tensor,
                            value: torch.Tensor,
                            reference_points: torch.Tensor,
                            spatial_shapes: torch.Tensor,
                            level_start_index: torch.Tensor,
                            indexes: list[torch.Tensor],
                            count: torch.Tensor,
                            num_query: int,
                            num_cams: int,
                            max_len: int) -> torch.Tensor:
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, query_len, _ = query.shape
        _, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, query_len, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, query_len, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(bs, query_len,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)

        if reference_points.shape[-1] != 2:
            raise ValueError('SCA direct scatter only supports 2D reference '
                             f'points, but got {reference_points.shape[-1]}.')
        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        bs, query_len, num_z_anchors, xy = reference_points.shape
        reference_points = reference_points[:, :, None, None, None, :, :]
        sampling_offsets = sampling_offsets / \
            offset_normalizer[None, None, None, :, None, :]
        _, _, num_heads, num_levels, num_all_points, _ = \
            sampling_offsets.shape
        sampling_offsets = sampling_offsets.view(
            bs, query_len, num_heads, num_levels,
            num_all_points // num_z_anchors, num_z_anchors, xy)
        sampling_locations = reference_points + sampling_offsets
        _, _, _, _, num_points, num_z_anchors, _ = sampling_locations.shape
        assert num_all_points == num_points * num_z_anchors
        sampling_locations = sampling_locations.view(
            bs, query_len, num_heads, num_levels, num_all_points, xy)

        return _SCAMSDeformAttnScatterDivFunction.apply(
            value, spatial_shapes, level_start_index, sampling_locations,
            attention_weights, indexes, count, num_query, num_cams, max_len,
            self.im2col_step)

    # def forward_bk(self,
    #             query,
    #             key=None,
    #             value=None,
    #             identity=None,
    #             query_pos=None,
    #             key_padding_mask=None,
    #             reference_points=None,
    #             spatial_shapes=None,
    #             level_start_index=None,
    #             **kwargs):
    #     """Forward Function of MultiScaleDeformAttention.
    #     Args:
    #         query (Tensor): Query of Transformer with shape
    #             ( bs, num_query, embed_dims).
    #         key (Tensor): The key tensor with shape
    #             `(bs, num_key,  embed_dims)`.
    #         value (Tensor): The value tensor with shape
    #             `(bs, num_key,  embed_dims)`.
    #         identity (Tensor): The tensor used for addition, with the
    #             same shape as `query`. Default None. If None,
    #             `query` will be used.
    #         query_pos (Tensor): The positional encoding for `query`.
    #             Default: None.
    #         key_pos (Tensor): The positional encoding for `key`. Default
    #             None.
    #         reference_points (Tensor):  The normalized reference
    #             points with shape (bs, num_query, num_levels, 2),
    #             all elements is range in [0, 1], top-left (0,0),
    #             bottom-right (1, 1), including padding area.
    #             or (N, Length_{query}, num_levels, 4), add
    #             additional two dimensions is (w, h) to
    #             form reference boxes.
    #         key_padding_mask (Tensor): ByteTensor for `query`, with
    #             shape [bs, num_key].
    #         spatial_shapes (Tensor): Spatial shape of features in
    #             different levels. With shape (num_levels, 2),
    #             last dimension represents (h, w).
    #         level_start_index (Tensor): The start index of each level.
    #             A tensor has shape ``(num_levels, )`` and can be represented
    #             as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
    #     Returns:
    #          Tensor: forwarded results with shape [num_query, bs, embed_dims].
    #     """

    #     if value is None:
    #         value = query
    #     if identity is None:
    #         identity = query
    #     if query_pos is not None:
    #         query = query + query_pos

    #     if not self.batch_first:
    #         # change to (bs, num_query ,embed_dims)
    #         query = query.permute(1, 0, 2)
    #         value = value.permute(1, 0, 2)

    #     bs, num_query, _ = query.shape
    #     bs, num_value, _ = value.shape
    #     assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

    #     value = self.value_proj(value)
    #     if key_padding_mask is not None:
    #         value = value.masked_fill(key_padding_mask[..., None], 0.0)
    #     value = value.view(bs, num_value, self.num_heads, -1)
    #     sampling_offsets = self.sampling_offsets(query).view(
    #         bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
    #     attention_weights = self.attention_weights(query).view(
    #         bs, num_query, self.num_heads, self.num_levels * self.num_points)

    #     attention_weights = attention_weights.softmax(-1)

    #     attention_weights = attention_weights.view(bs, num_query,
    #                                                self.num_heads,
    #                                                self.num_levels,
    #                                                self.num_points)

    #     if reference_points.shape[-1] == 2:
    #         """
    #         For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
    #         After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
    #         For each referent point, we sample `num_points` sampling points.
    #         For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
    #         """
    #         offset_normalizer = torch.stack(
    #             [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

    #         bs, num_query, num_Z_anchors, xy = reference_points.shape
    #         reference_points = reference_points[:, :, None, None, None, :, :]
    #         sampling_offsets = sampling_offsets / \
    #             offset_normalizer[None, None, None, :, None, :]
    #         bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
    #         sampling_offsets = sampling_offsets.view(
    #             bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)
    #         sampling_locations = reference_points + sampling_offsets
    #         bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
    #         assert num_all_points == num_points * num_Z_anchors

    #         sampling_locations = sampling_locations.view(
    #             bs, num_query, num_heads, num_levels, num_all_points, xy)

    #     elif reference_points.shape[-1] == 4:
    #         assert False
    #     else:
    #         raise ValueError(
    #             f'Last dim of reference_points must be'
    #             f' 2 or 4, but get {reference_points.shape[-1]} instead.')

    #     #  sampling_locations.shape: bs, num_query, num_heads, num_levels, num_all_points, 2
    #     #  attention_weights.shape: bs, num_query, num_heads, num_levels, num_all_points
    #     #

    #     if torch.musa.is_available() and value.is_musa:
    #         if value.dtype == torch.float16:
    #             MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
    #         else:
    #             MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
    #         output = MultiScaleDeformableAttnFunction.apply(
    #             value, spatial_shapes, level_start_index, sampling_locations,
    #             attention_weights, self.im2col_step)
    #     else:
    #         output = multi_scale_deformable_attn_pytorch(
    #             value, spatial_shapes, sampling_locations, attention_weights)
    #     if not self.batch_first:
    #         output = output.permute(1, 0, 2)

    #     return output

    def forward(self,
                query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                identity: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                reference_points: Optional[torch.Tensor] = None,
                spatial_shapes: Optional[torch.Tensor] = None,
                level_start_index: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        del key, identity, kwargs
        if value is None:
            value = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        _, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            bs, num_query, num_z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, None, :, :]
            sampling_offsets = sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]
            _, _, num_heads, num_levels, num_all_points, _ = \
                sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels,
                num_all_points // num_z_anchors, num_z_anchors, xy)
            sampling_locations = reference_points + sampling_offsets
            _, _, _, _, num_points, num_z_anchors, _ = \
                sampling_locations.shape
            assert num_all_points == num_points * num_z_anchors
            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)
        elif reference_points.shape[-1] == 4:
            raise AssertionError(
                'MSDeformableAttention3D does not support 4D reference '
                'points in this integration.')
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        output = _multi_scale_deform_attn(value, spatial_shapes,
                                          level_start_index,
                                          sampling_locations,
                                          attention_weights,
                                          self.im2col_step)
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        return output
