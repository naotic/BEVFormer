import os
import json
import torch
import torch.distributed as dist


def _get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _tensor_meta(x):
    if not torch.is_tensor(x):
        return {
            "type": type(x).__name__,
            "repr": repr(x),
        }

    return {
        "type": "Tensor",
        "shape": tuple(x.shape),
        "stride": tuple(x.stride()),
        "is_contiguous": x.is_contiguous(),
        "is_channels_last": (
            x.is_contiguous(memory_format=torch.channels_last)
            if x.dim() == 4 else False
        ),
        "dtype": str(x.dtype),
        "device": str(x.device),
        "requires_grad": x.requires_grad,
    }


def _safe_clone_for_save(x, cpu=True):
    if torch.is_tensor(x):
        # 保存输入本体，避免后续 inplace 修改影响 dump 内容
        y = x.detach().clone()
        if cpu:
            y = y.cpu()
        return y
    elif isinstance(x, (list, tuple)):
        return type(x)(_safe_clone_for_save(v, cpu=cpu) for v in x)
    elif isinstance(x, dict):
        return {k: _safe_clone_for_save(v, cpu=cpu) for k, v in x.items()}
    else:
        return x


def dump_forward_inputs_once(tag, inputs_dict, save_dir="/tmp/bevformer_forward_dump", cpu=True):
    """
    tag: 用于区分不同 forward，例如 'ms_deform_attn' / 'spatial_cross_attn'
    inputs_dict: {'query': query, 'key': key, ...}
    cpu: True 表示保存到 CPU，文件更通用；False 表示保留 MUSA tensor
    """
    rank = _get_rank()

    # 多卡建议只保存 rank0，避免每张卡都保存一份。
    # 如果你想每个 rank 都保存，可以删掉这个判断。
    if rank != 0:
        return

    os.makedirs(save_dir, exist_ok=True)

    flag_file = os.path.join(save_dir, f"{tag}.done")
    if os.path.exists(flag_file):
        return

    meta = {}
    save_obj = {}

    print(f"\n========== dump forward inputs: {tag} ==========")

    for name, value in inputs_dict.items():
        meta[name] = _tensor_meta(value)
        save_obj[name] = _safe_clone_for_save(value, cpu=cpu)

        if torch.is_tensor(value):
            print(
                f"{name}: "
                f"shape={tuple(value.shape)}, "
                f"stride={tuple(value.stride())}, "
                f"contig={value.is_contiguous()}, "
                f"dtype={value.dtype}, "
                f"device={value.device}, "
                f"requires_grad={value.requires_grad}"
            )
        else:
            print(f"{name}: type={type(value).__name__}, value={repr(value)}")

    pt_path = os.path.join(save_dir, f"{tag}_inputs.pt")
    meta_path = os.path.join(save_dir, f"{tag}_meta.json")

    torch.save(save_obj, pt_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    with open(flag_file, "w") as f:
        f.write("done\n")

    print(f"[dump] saved tensor inputs to: {pt_path}")
    print(f"[dump] saved meta to: {meta_path}")
    print(f"========== dump forward inputs done: {tag} ==========\n")
