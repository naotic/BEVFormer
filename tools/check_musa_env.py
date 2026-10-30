#!/usr/bin/env python3
"""Inspect the delivered MUSA environment without allocating tensors or devices."""
import importlib
import importlib.machinery
import importlib.metadata
import os
from pathlib import Path
import sys

# Do not write bytecode, import train.py, build a model, or query device availability.
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("BEVFORMER_SELECTED_CONFIG", "1")


def main():
    failures = []
    loaded = {}
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Checkout: {ROOT}")
    print(f"Inherited PYTHONPATH: {os.environ.get('PYTHONPATH', '<unset>')}")
    print(f"Effective sys.path[0]: {sys.path[0]}")
    print(f"BEVFORMER_SELECTED_CONFIG: {os.environ['BEVFORMER_SELECTED_CONFIG']}")

    packages = (
        ("torch", "torch"),
        ("torch_musa", "torch_musa"),
        ("mmcv", "mmcv-full"),
        ("mmdet", "mmdet"),
        ("mmdet3d", "mmdet3d"),
        ("mmdet3d.ops", "mmdet3d"),
        ("mmseg", "mmsegmentation"),
        ("modulated_deform_conv", None),
        ("modulated_deform_conv.perf", None),
        ("_modulated_deform_conv_ext", None),
        ("msda_musa_opt2", None),
        ("_msda_opt2_musa", None),
        ("bevformer_musa_fusion", "bevformer-musa-fusion"),
    )
    for name, distribution in packages:
        try:
            module = importlib.import_module(name)
            loaded[name] = module
            version = getattr(module, "__version__", "<not exposed>")
            try:
                dist_version = importlib.metadata.version(distribution) if distribution else "<not queried>"
            except importlib.metadata.PackageNotFoundError:
                dist_version = "<metadata unavailable>"
            print(f"[OK] {name}: module={version}; distribution={dist_version}; source={module.__file__}")
        except Exception as exc:
            failures.append(name)
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")

    required_symbols = {
        "modulated_deform_conv": ("ModulatedDeformConv2dPack",),
        "modulated_deform_conv.perf": ("patch_1x1_conv",),
        "_msda_opt2_musa": ("forward", "backward"),
        "_modulated_deform_conv_ext": (
            "modulated_deformable_im2col_nhwc_cuda",
            "modulated_deformable_fused_im2col_nhwc_cuda",
            "modulated_deformable_col2im_nhwc_cuda",
            "modulated_deformable_col2im_coord_nhwc_cuda",
            "modulated_deformable_col2im_perelem_nhwc_cuda",
            "modulated_deformable_col2im_perelem_smem_nhwc_cuda",
        ),
    }
    for name, symbols in required_symbols.items():
        module = loaded.get(name)
        missing = [symbol for symbol in symbols if not callable(getattr(module, symbol, None))]
        if missing:
            failures.append(name + " entry points")
            print(f"[FAIL] {name} missing entry points: {', '.join(missing)}")
        else:
            print(f"[OK] {name} entry points: {', '.join(symbols)}")

    # Resolve the plugin without executing its model imports or device probes.
    try:
        projects = importlib.import_module("projects")
        if Path(projects.__file__).resolve() != ROOT / "projects" / "__init__.py":
            raise RuntimeError(f"projects resolves outside this checkout: {projects.__file__}")
        spec = importlib.machinery.PathFinder.find_spec(
            "projects.mmdet3d_plugin", projects.__path__)
        expected = ROOT / "projects" / "mmdet3d_plugin" / "__init__.py"
        if spec is None or spec.origin is None or Path(spec.origin).resolve() != expected:
            raise RuntimeError(f"unexpected plugin origin: {getattr(spec, 'origin', None)}")
        print(f"[OK] projects: {projects.__file__}")
        print(f"[OK] plugin import origin (not executed): {spec.origin}")
    except Exception as exc:
        failures.append("checkout import")
        print(f"[FAIL] checkout import: {type(exc).__name__}: {exc}")

    try:
        ext = importlib.import_module("mmcv._ext")
        for symbol in ("ms_deform_attn_forward", "ms_deform_attn_backward"):
            if not callable(getattr(ext, symbol, None)):
                raise RuntimeError(f"missing mmcv._ext.{symbol}")
        print(f"[OK] MMCV MSDA entry points: {ext.__file__}")
    except Exception as exc:
        failures.append("MMCV MSDA")
        print(f"[FAIL] MMCV MSDA: {type(exc).__name__}: {exc}")

    try:
        torch = loaded["torch"]
        op = torch.ops.musa_fused.conv2d_bias_act_add.default
        print(f"Fusion schema: {op._schema}")
        registered = torch._C._dispatch_has_kernel_for_dispatch_key(
            "musa_fused::conv2d_bias_act_add", "PrivateUse1")
        if not registered:
            raise RuntimeError("fusion has no PrivateUse1 kernel")
        print("[OK] fusion PrivateUse1 kernel registered")
    except Exception as exc:
        failures.append("fusion registration")
        print(f"[FAIL] fusion registration: {type(exc).__name__}: {exc}")

    if failures:
        print("Environment check failed: " + ", ".join(failures))
        print("Use the delivered image and inspect the errors; do not replace its packages with pip.")
        return 1
    print("Import/registration checks passed. No model, GPU allocation, or training was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
