import importlib
import os
from typing import Callable, Optional


_LOGGED = False


def _log_once(msg: str) -> None:
    global _LOGGED
    if _LOGGED:
        return
    _LOGGED = True
    print(msg)


def _get_env_flag(name: str, default: str = "1") -> bool:
    v = os.getenv(name, default).strip().lower()
    return v not in ("0", "false", "off", "no", "")


def try_enable_msda_opt() -> bool:
    """Try to replace BEVFormer MSDA autograd.Function with optimized MUSA extension.

    Scope: BEVFormer only (monkeypatch the module
    `bevformer.modules.multi_scale_deformable_attn_function`).

    Returns:
        True if replacement succeeded, else False (fallback to original).
    """
    if not _get_env_flag("MSDA_OPT_ENABLE", "1"):
        return False

    try:
        import torch

        if not hasattr(torch, "musa") or not torch.musa.is_available():
            return False
    except Exception:
        return False

    try:
        opt_mod = importlib.import_module("_msda_opt_musa")
        opt_forward: Optional[Callable] = getattr(opt_mod, "forward", None)
        opt_backward: Optional[Callable] = getattr(opt_mod, "backward", None)
        if opt_forward is None or opt_backward is None:
            return False
    except Exception:
        return False

    try:
        target = importlib.import_module(
            "projects.mmdet3d_plugin.bevformer.modules.multi_scale_deformable_attn_function"
        )
    except Exception:
        # Some setups import as `mmdet3d_plugin...`
        try:
            target = importlib.import_module(
                "mmdet3d_plugin.bevformer.modules.multi_scale_deformable_attn_function"
            )
        except Exception:
            return False

    def _patch_cls(cls) -> bool:
        if cls is None:
            return False
        if not hasattr(cls, "forward") or not hasattr(cls, "backward"):
            return False
        try:
            setattr(cls, "_msda_opt_prev_forward", cls.forward)
            setattr(cls, "_msda_opt_prev_backward", cls.backward)
            setattr(cls, "forward", staticmethod(opt_forward))
            setattr(cls, "backward", staticmethod(opt_backward))
            return True
        except Exception:
            return False

    ok_fp32 = _patch_cls(getattr(target, "MultiScaleDeformableAttnFunction_fp32", None))
    ok_fp16 = _patch_cls(getattr(target, "MultiScaleDeformableAttnFunction_fp16", None))

    if ok_fp32 or ok_fp16:
        _log_once("[msda_opt] Replaced BEVFormer MSDA with optimized implementation")
        return True
    return False
