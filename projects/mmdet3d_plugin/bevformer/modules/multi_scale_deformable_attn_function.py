# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.autograd.function import Function, once_differentiable
from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', [
        'ms_deform_attn_backward', 'ms_deform_attn_forward',
        'sca_ms_deform_attn_scatter_div_forward'
    ])

import os as _os_msda

# MSDA col2im smem-reduce backward (opt: msda_col2im_smem / env MSDA_COL2IM_SMEM_REDUCE).
# When enabled, route the MSDA backward (grad_value col2im + grad sampling/attn) through
# the msda_musa_opt2 kernel (force_generic_col2im=False); that kernel reads
# MSDA_COL2IM_SMEM_REDUCE internally to pick the 256-thread / shared-memory-reduce
# col2im (~1.94x on the backward). Default OFF -> the baseline ext_module in-place
# backward (byte-identical to current bevformer_bk). Math is identical either way.
try:
    from msda_musa_opt2.dispatch import ms_deform_attn_backward as _opt2_msda_backward
except Exception:  # package absent / non-MUSA env -> always fall back to ext_module
    _opt2_msda_backward = None


def _col2im_smem_enabled():
    return (_opt2_msda_backward is not None
            and _os_msda.environ.get("MSDA_COL2IM_SMEM_REDUCE", "0").lower()
            in ("1", "true", "yes", "on"))


def _msda_backward_grads(value, spatial_shapes, level_start_index,
                         sampling_locations, attention_weights, grad_output,
                         im2col_step):
    """Return (grad_value, grad_sampling_loc, grad_attn_weight) for the standard
    MSDA backward. Uses the msda_musa_opt2 smem col2im kernel when enabled and
    the value is on musa; otherwise the baseline ext_module in-place backward."""
    if _col2im_smem_enabled() and value.device.type == 'musa':
        return _opt2_msda_backward(
            value, spatial_shapes, level_start_index, sampling_locations,
            attention_weights, grad_output.contiguous(),
            im2col_step=im2col_step, force_generic_col2im=False)
    grad_value = torch.zeros_like(value)
    grad_sampling_loc = torch.zeros_like(sampling_locations)
    grad_attn_weight = torch.zeros_like(attention_weights)
    ext_module.ms_deform_attn_backward(
        value, spatial_shapes, level_start_index, sampling_locations,
        attention_weights, grad_output.contiguous(),
        grad_value, grad_sampling_loc, grad_attn_weight,
        im2col_step=im2col_step)
    return grad_value, grad_sampling_loc, grad_attn_weight

def _get_sca_mask_metadata(
        bev_mask: torch.Tensor) -> tuple[list[torch.Tensor], tuple[int, ...], int,
                                         torch.Tensor]:
    bev_mask_any = bev_mask.sum(-1) > 0
    indexes = []
    valid_lens = []
    for cam_mask in bev_mask_any[:, 0]:
        index_query_per_img = cam_mask.nonzero().squeeze(-1)
        indexes.append(index_query_per_img)
        valid_lens.append(int(index_query_per_img.numel()))
    count = torch.clamp(bev_mask_any.permute(1, 2, 0).sum(-1), min=1)
    return indexes, tuple(valid_lens), max(valid_lens, default=0), count


class _SCAAttentionScatterFunction(Function):

    @staticmethod
    def forward(ctx, queries: torch.Tensor, indexes: list[torch.Tensor],
                num_query: int) -> torch.Tensor:
        ctx.indexes = indexes
        ctx.num_cams = queries.size(1)
        ctx.max_len = queries.size(2)
        return ext_module.sca_attention_scatter_forward(queries, indexes,
                                                        num_query)

    @staticmethod
    def backward(ctx, grad_slots: torch.Tensor):
        grad_queries = ext_module.sca_attention_scatter_backward(
            grad_slots.contiguous(), ctx.indexes, ctx.num_cams, ctx.max_len)
        return grad_queries, None, None



class _SCAAttentionScatterDivFunction(Function):

    @staticmethod
    def forward(ctx, queries: torch.Tensor, indexes: list[torch.Tensor],
                count: torch.Tensor, num_query: int) -> torch.Tensor:
        count = count.to(dtype=queries.dtype)
        ctx.indexes = indexes
        ctx.count = count
        ctx.num_cams = queries.size(1)
        ctx.max_len = queries.size(2)
        return ext_module.sca_attention_scatter_div_forward(
            queries, indexes, count, num_query)

    @staticmethod
    def backward(ctx, grad_slots: torch.Tensor):
        grad_queries = ext_module.sca_attention_scatter_div_backward(
            grad_slots.contiguous(), ctx.indexes, ctx.count, ctx.num_cams,
            ctx.max_len)
        return grad_queries, None, None, None


class _SCAMSDeformAttnScatterDivFunction(Function):

    @staticmethod
    def forward(ctx, value: torch.Tensor, spatial_shapes: torch.Tensor,
                level_start_index: torch.Tensor,
                sampling_locations: torch.Tensor,
                attention_weights: torch.Tensor, indexes: list[torch.Tensor],
                count: torch.Tensor, num_query: int, num_cams: int,
                max_len: int, im2col_step: int) -> torch.Tensor:
        ctx.value_dtype = value.dtype
        ctx.sampling_dtype = sampling_locations.dtype
        ctx.attention_dtype = attention_weights.dtype
        ctx.indexes = indexes
        ctx.count = count.float().contiguous()
        ctx.num_cams = num_cams
        ctx.max_len = max_len
        ctx.im2col_step = im2col_step

        value_fp32 = value.float().contiguous()
        sampling_locations_fp32 = sampling_locations.float().contiguous()
        attention_weights_fp32 = attention_weights.float().contiguous()
        output = ext_module.sca_ms_deform_attn_scatter_div_forward(
            value_fp32, spatial_shapes, level_start_index,
            sampling_locations_fp32, attention_weights_fp32, indexes,
            ctx.count, num_query)
        ctx.save_for_backward(value_fp32, spatial_shapes, level_start_index,
                              sampling_locations_fp32,
                              attention_weights_fp32)
        return output.to(ctx.value_dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_slots: torch.Tensor):
        value, spatial_shapes, level_start_index, sampling_locations, \
            attention_weights = ctx.saved_tensors
        grad_queries = ext_module.sca_attention_scatter_div_backward(
            grad_slots.float().contiguous(), ctx.indexes, ctx.count,
            ctx.num_cams, ctx.max_len)
        grad_output = grad_queries.view(-1, ctx.max_len, grad_queries.size(-1))
        grad_value, grad_sampling_loc, grad_attn_weight = _msda_backward_grads(
            value, spatial_shapes, level_start_index,
            sampling_locations, attention_weights, grad_output,
            ctx.im2col_step)

        return grad_value.to(ctx.value_dtype), None, None, \
            grad_sampling_loc.to(ctx.sampling_dtype), \
            grad_attn_weight.to(ctx.attention_dtype), None, None, None, \
            None, None, None


def _sca_python_scatter(queries: torch.Tensor, indexes: list[torch.Tensor],
                        valid_lens: tuple[int, ...],
                        num_query: int) -> torch.Tensor:
    slots = queries.new_zeros((queries.size(0), num_query, queries.size(3)))
    for batch_id in range(queries.size(0)):
        for cam_id, index_query_per_img in enumerate(indexes):
            valid_len = valid_lens[cam_id]
            if valid_len == 0:
                continue
            slots[batch_id, index_query_per_img] += \
                queries[batch_id, cam_id, :valid_len]
    return slots


class _SCAQueryGatherFunction(Function):

    @staticmethod
    def forward(ctx, query: torch.Tensor, indexes: list[torch.Tensor],
                num_cams: int, max_len: int) -> torch.Tensor:
        ctx.indexes = indexes
        ctx.num_query = query.size(1)
        return ext_module.sca_query_gather_forward(query, indexes, num_cams,
                                                   max_len)

    @staticmethod
    def backward(ctx, grad_queries_rebatch: torch.Tensor):
        grad_query = ext_module.sca_query_gather_backward(
            grad_queries_rebatch.contiguous(), ctx.indexes, ctx.num_query)
        return grad_query, None, None, None


class _SCAQueryReferenceGatherFunction(Function):

    @staticmethod
    def forward(ctx, query: torch.Tensor, reference_points_cam: torch.Tensor,
                indexes: list[torch.Tensor], num_cams: int,
                max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.indexes = indexes
        ctx.num_query = query.size(1)
        outputs = ext_module.sca_query_reference_gather_forward(
            query, reference_points_cam, indexes, num_cams, max_len)
        return outputs[0], outputs[1]

    @staticmethod
    def backward(ctx, grad_queries_rebatch: torch.Tensor,
                 grad_reference_points_rebatch: torch.Tensor):
        grad_query = None
        grad_reference_points_cam = None
        if grad_queries_rebatch is not None:
            grad_query = ext_module.sca_query_gather_backward(
                grad_queries_rebatch.contiguous(), ctx.indexes,
                ctx.num_query)
        if grad_reference_points_rebatch is not None:
            grad_reference_points_cam = ext_module.sca_reference_gather_backward(
                grad_reference_points_rebatch.contiguous(), ctx.indexes,
                ctx.num_query)
        return grad_query, grad_reference_points_cam, None, None, None

def multi_scale_deformable_attn_pytorch(
        value: torch.Tensor, value_spatial_shapes: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor) -> torch.Tensor:
    """CPU version of multi-scale deformable attention.

    Args:
        value (torch.Tensor): The value has shape
            (bs, num_keys, num_heads, embed_dims//num_heads)
        value_spatial_shapes (torch.Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (torch.Tensor): The location of sampling points,
            has shape
            (bs ,num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (torch.Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points),

    Returns:
        torch.Tensor: has shape (bs, num_queries, embed_dims)
    """

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ =\
        sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes],
                             dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
            bs * num_heads, embed_dims, H_, W_)
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :,
                                          level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).view(bs, num_heads * embed_dims,
                                              num_queries)
    return output.transpose(1, 2).contiguous()


def _can_use_ms_deform_attn_ext(value: torch.Tensor) -> bool:
    # return ((IS_CUDA_AVAILABLE and value.is_cuda)
    #         or (IS_MLU_AVAILABLE and value.is_mlu)
    #         or (IS_MUSA_AVAILABLE and value.device.type == 'musa')
    #         or (IS_NPU_AVAILABLE and value.device.type == 'npu'))
    return torch.musa.is_available() and value.device.type == 'musa'

def _multi_scale_deform_attn(value: torch.Tensor,
                             spatial_shapes: torch.Tensor,
                             level_start_index: torch.Tensor,
                             sampling_locations: torch.Tensor,
                             attention_weights: torch.Tensor,
                             im2col_step: int) -> torch.Tensor:
    if not _can_use_ms_deform_attn_ext(value):
        return multi_scale_deformable_attn_pytorch(value, spatial_shapes,
                                                   sampling_locations,
                                                   attention_weights)
    if torch.musa.is_available() and value.device.type == 'musa':
        return MultiScaleDeformableAttnFunction_fp32.apply(
            value, spatial_shapes, level_start_index, sampling_locations,
            attention_weights, im2col_step)
    return MultiScaleDeformableAttnFunction.apply(value, spatial_shapes,
                                                  level_start_index,
                                                  sampling_locations,
                                                  attention_weights,
                                                  im2col_step)


class MultiScaleDeformableAttnFunction(Function):

    @staticmethod
    def forward(ctx, value: torch.Tensor, value_spatial_shapes: torch.Tensor,
                value_level_start_index: torch.Tensor,
                sampling_locations: torch.Tensor,
                attention_weights: torch.Tensor,
                im2col_step: torch.Tensor) -> torch.Tensor:
        """GPU/MLU version of multi-scale deformable attention.

        Args:
            value (torch.Tensor): The value has shape
                (bs, num_keys, mum_heads, embed_dims//num_heads)
            value_spatial_shapes (torch.Tensor): Spatial shape of
                each feature map, has shape (num_levels, 2),
                last dimension 2 represent (h, w)
            sampling_locations (torch.Tensor): The location of sampling points,
                has shape
                (bs ,num_queries, num_heads, num_levels, num_points, 2),
                the last dimension 2 represent (x, y).
            attention_weights (torch.Tensor): The weight of sampling points
                used when calculate the attention, has shape
                (bs ,num_queries, num_heads, num_levels, num_points),
            im2col_step (torch.Tensor): The step used in image to column.

        Returns:
            torch.Tensor: has shape (bs, num_queries, embed_dims)
        """

        ctx.im2col_step = im2col_step

        # When pytorch version >= 1.6.0, amp is adopted for fp16 mode;
        # amp won't cast the type of sampling_locations, attention_weights
        # (float32), but "value" is cast to float16, leading to the type
        # mismatch with input (when it is float32) or weight.
        # The flag for whether to use fp16 or amp is the type of "value",
        # we cast sampling_locations and attention_weights to
        # temporarily support fp16 and amp whatever the
        # pytorch version is.
        value = value.contiguous()
        sampling_locations = sampling_locations.type_as(value).contiguous()
        attention_weights = attention_weights.type_as(value).contiguous()

        output = ext_module.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step=ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations,
                              attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """GPU/MLU version of backward function.

        Args:
            grad_output (torch.Tensor): Gradient of output tensor of forward.

        Returns:
            tuple[Tensor]: Gradient of input tensors in forward.
        """
        value, value_spatial_shapes, value_level_start_index, \
            sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = _msda_backward_grads(
            value, value_spatial_shapes, value_level_start_index,
            sampling_locations, attention_weights, grad_output,
            ctx.im2col_step)

        return grad_value, None, None, \
            grad_sampling_loc, grad_attn_weight, None

# from msda_musa_opt2.dispatch import ms_deform_attn_backward, ms_deform_attn_forward

import os

SAVE_DIR = "./dump_ms_deform_attn"
os.makedirs(SAVE_DIR, exist_ok=True)

_dump_counter = {
    "fwd": 0,
    "bwd": 0
}

def _tensor_meta(x):
    return {
        "shape": tuple(x.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
        "stride": tuple(x.stride()),
        "is_contiguous": x.is_contiguous(),
    }

def _dump(name, data):
    path = os.path.join(SAVE_DIR, f"{name}.pt")
    torch.save(data, path)
    print(f"[DUMP] saved to {path}")


class MultiScaleDeformableAttnFunction_fp16(Function):

    @staticmethod
    # @custom_fwd(cast_inputs=torch.float16)
    @torch.musa.amp.custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        """GPU version of multi-scale deformable attention.

        Args:
            value (Tensor): The value has shape
                (bs, num_keys, mum_heads, embed_dims//num_heads)
            value_spatial_shapes (Tensor): Spatial shape of
                each feature map, has shape (num_levels, 2),
                last dimension 2 represent (h, w)
            sampling_locations (Tensor): The location of sampling points,
                has shape
                (bs ,num_queries, num_heads, num_levels, num_points, 2),
                the last dimension 2 represent (x, y).
            attention_weights (Tensor): The weight of sampling points used
                when calculate the attention, has shape
                (bs ,num_queries, num_heads, num_levels, num_points),
            im2col_step (Tensor): The step used in image to column.

        Returns:
            Tensor: has shape (bs, num_queries, embed_dims)
        """
        ctx.im2col_step = im2col_step
        output = ext_module.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step=ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations,
                              attention_weights)
        return output

    @staticmethod
    @once_differentiable
    # @custom_bwd
    @torch.musa.amp.custom_bwd
    def backward(ctx, grad_output):
        """GPU version of backward function.

        Args:
            grad_output (Tensor): Gradient
                of output tensor of forward.

        Returns:
             Tuple[Tensor]: Gradient
                of input tensors in forward.
        """
        value, value_spatial_shapes, value_level_start_index, \
            sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = _msda_backward_grads(
            value, value_spatial_shapes, value_level_start_index,
            sampling_locations, attention_weights, grad_output,
            ctx.im2col_step)

        return grad_value, None, None, \
            grad_sampling_loc, grad_attn_weight, None


class MultiScaleDeformableAttnFunction_fp32(Function):

    @staticmethod
    # @custom_fwd(cast_inputs=torch.float32)
    @torch.musa.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        """GPU version of multi-scale deformable attention.

        Args:
            value (Tensor): The value has shape
                (bs, num_keys, mum_heads, embed_dims//num_heads)
            value_spatial_shapes (Tensor): Spatial shape of
                each feature map, has shape (num_levels, 2),
                last dimension 2 represent (h, w)
            sampling_locations (Tensor): The location of sampling points,
                has shape
                (bs ,num_queries, num_heads, num_levels, num_points, 2),
                the last dimension 2 represent (x, y).
            attention_weights (Tensor): The weight of sampling points used
                when calculate the attention, has shape
                (bs ,num_queries, num_heads, num_levels, num_points),
            im2col_step (Tensor): The step used in image to column.

        Returns:
            Tensor: has shape (bs, num_queries, embed_dims)
        """
        # print(f"value shape: {value.shape}")
        # print(f"value_spatial_shapes shape: {value_spatial_shapes.shape}")
        # print(f"value_level_start_index shape: {value_level_start_index.shape}")
        # print(f"sampling_locations shape: {sampling_locations.shape}")
        # print(f"attention_weights shape: {attention_weights.shape}")
        # print(f"im2col_step: {im2col_step}")
        # raise NotImplementedError("MultiScaleDeformableAttnFunction_fp32 is not implemented yet.")

        ctx.im2col_step = im2col_step
        # if _dump_counter["fwd"] < 1:   # 只抓1次，你可以改成 <3
        #     dump_data = {
        #         "op": "ms_deform_attn_forward",
        #         "inputs": {
        #             "value": {
        #                 "tensor": value.detach().cpu(),
        #                 "meta": _tensor_meta(value),
        #             },
        #             "value_spatial_shapes": {
        #                 "tensor": value_spatial_shapes.detach().cpu(),
        #                 "meta": _tensor_meta(value_spatial_shapes),
        #             },
        #             "value_level_start_index": {
        #                 "tensor": value_level_start_index.detach().cpu(),
        #                 "meta": _tensor_meta(value_level_start_index),
        #             },
        #             "sampling_locations": {
        #                 "tensor": sampling_locations.detach().cpu(),
        #                 "meta": _tensor_meta(sampling_locations),
        #             },
        #             "attention_weights": {
        #                 "tensor": attention_weights.detach().cpu(),
        #                 "meta": _tensor_meta(attention_weights),
        #             },
        #         },
        #         "scalars": {
        #             "im2col_step": int(im2col_step),
        #         }
        #     }

        #     _dump(f"ms_deform_attn_forward{_dump_counter['fwd']}", dump_data)
        #     _dump_counter["fwd"] += 1
        output = ext_module.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step=ctx.im2col_step)
        # output = ms_deform_attn_forward(
        #     value,
        #     value_spatial_shapes,
        #     value_level_start_index,
        #     sampling_locations,
        #     attention_weights,
        #     im2col_step=ctx.im2col_step,
        #     force_generic=False)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations,
                              attention_weights)
        return output

    @staticmethod
    @once_differentiable
    # @custom_bwd
    @torch.musa.amp.custom_bwd
    def backward(ctx, grad_output):
        """GPU version of backward function.

        Args:
            grad_output (Tensor): Gradient
                of output tensor of forward.

        Returns:
             Tuple[Tensor]: Gradient
                of input tensors in forward.
        """
        value, value_spatial_shapes, value_level_start_index, \
            sampling_locations, attention_weights = ctx.saved_tensors
        # if _dump_counter["bwd"] < 1:
        #     dump_data = {
        #         "op": "ms_deform_attn_backward",
        #         "inputs": {
        #             "value": {
        #                 "tensor": value.detach().cpu(),
        #                 "meta": _tensor_meta(value),
        #             },
        #             "value_spatial_shapes": {
        #                 "tensor": value_spatial_shapes.detach().cpu(),
        #                 "meta": _tensor_meta(value_spatial_shapes),
        #             },
        #             "value_level_start_index": {
        #                 "tensor": value_level_start_index.detach().cpu(),
        #                 "meta": _tensor_meta(value_level_start_index),
        #             },
        #             "sampling_locations": {
        #                 "tensor": sampling_locations.detach().cpu(),
        #                 "meta": _tensor_meta(sampling_locations),
        #             },
        #             "attention_weights": {
        #                 "tensor": attention_weights.detach().cpu(),
        #                 "meta": _tensor_meta(attention_weights),
        #             },
        #             "grad_output": {
        #                 "tensor": grad_output.detach().cpu(),
        #                 "meta": _tensor_meta(grad_output),
        #             },
        #         },
        #         "scalars": {
        #             "im2col_step": int(ctx.im2col_step),
        #         }
        #     }

        #     _dump(f"ms_deform_attn_backward{_dump_counter['bwd']}", dump_data)
        #     _dump_counter["bwd"] += 1
        grad_value, grad_sampling_loc, grad_attn_weight = _msda_backward_grads(
            value, value_spatial_shapes, value_level_start_index,
            sampling_locations, attention_weights, grad_output,
            ctx.im2col_step)
        # grad_value, grad_sampling_loc, grad_attn_weight = ms_deform_attn_backward(
        #     value,
        #     value_spatial_shapes,
        #     value_level_start_index,
        #     sampling_locations,
        #     attention_weights,
        #     grad_output,
        #     im2col_step=ctx.im2col_step,
        #     force_generic_col2im=False,
        # )
        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None
