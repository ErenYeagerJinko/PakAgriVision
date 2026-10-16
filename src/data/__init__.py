"""PakAgriVision data processing package."""

from src.data.quality import (
    QualityChecker,
    compute_md5,
    compute_phash,
    scan_image_files,
    verify_image,
)
from src.data.splits import (
    compute_class_weights,
    create_stratified_group_splits,
    get_class_mappings,
    load_class_names,
)

try:
    from src.data.dataset import (
        FieldDataset,
        PlantDataset,
        create_weighted_sampler,
        get_dataloaders,
        seed_worker,
    )
    from src.data.transforms import (
        denormalize_tensor,
        get_eval_transforms,
        get_train_transforms,
    )
except ImportError:
    pass

__all__ = [
    "QualityChecker",
    "compute_md5",
    "compute_phash",
    "scan_image_files",
    "verify_image",
    "compute_class_weights",
    "create_stratified_group_splits",
    "get_class_mappings",
    "load_class_names",
    "PlantDataset",
    "FieldDataset",
    "create_weighted_sampler",
    "get_dataloaders",
    "seed_worker",
    "get_train_transforms",
    "get_eval_transforms",
    "denormalize_tensor",
]
