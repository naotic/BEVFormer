"""Tensor-only camera projection for the optional compiled encoder path."""
import torch


@torch.compile(fullgraph=True)
def project_points(reference_points, lidar2img, pc_range,
                   image_height, image_width):
    reference_points = reference_points.clone()
    for axis in range(3):
        reference_points[..., axis:axis + 1] = (
            reference_points[..., axis:axis + 1]
            * (pc_range[axis + 3] - pc_range[axis]) + pc_range[axis])
    reference_points = torch.cat(
        (reference_points, torch.ones_like(reference_points[..., :1])), -1)
    reference_points = reference_points.permute(1, 0, 2, 3)
    depth, batch, queries = reference_points.shape[:3]
    cameras = lidar2img.size(1)
    reference_points = reference_points.view(
        depth, batch, 1, queries, 4).repeat(1, 1, cameras, 1, 1).unsqueeze(-1)
    lidar2img = lidar2img.view(
        1, batch, cameras, 1, 4, 4).repeat(depth, 1, 1, queries, 1, 1)
    projected = torch.matmul(lidar2img.float(), reference_points.float()).squeeze(-1)
    mask = projected[..., 2:3] > 1e-5
    projected = projected[..., :2] / torch.maximum(
        projected[..., 2:3], torch.ones_like(projected[..., 2:3]) * 1e-5)
    projected[..., 0] /= image_width
    projected[..., 1] /= image_height
    mask = (mask & (projected[..., 1:2] > 0) & (projected[..., 1:2] < 1)
            & (projected[..., 0:1] < 1) & (projected[..., 0:1] > 0))
    mask = torch.nan_to_num(mask)
    return projected.permute(2, 1, 3, 0, 4), mask.permute(2, 1, 3, 0, 4).squeeze(-1)
