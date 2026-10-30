from .nuscenes_dataset import CustomNuScenesDataset
if __import__("os").environ.get("BEVFORMER_SELECTED_CONFIG") != "1":
    from .nuscenes_dataset_v2 import CustomNuScenesDatasetV2

from .builder import custom_build_dataset
__all__ = [
    'CustomNuScenesDataset',
    'CustomNuScenesDatasetV2',
]

if __import__("os").environ.get("BEVFORMER_SELECTED_CONFIG") == "1":
    __all__ = [name for name in __all__ if name != "CustomNuScenesDatasetV2"]
