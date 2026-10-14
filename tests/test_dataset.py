"""Unit tests for PyTorch Dataset, transform pipelines, and DataLoader construction."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.data.dataset import (
    FieldDataset,
    PlantDataset,
    create_weighted_sampler,
    get_dataloaders,
)
from src.data.transforms import (
    denormalize_tensor,
    get_eval_transforms,
    get_train_transforms,
)


@pytest.fixture
def mock_image_environment(tmp_path: Path):
    """Create synthetic image files and mock split DataFrames for testing."""
    root_dir = tmp_path / "synthetic_raw"
    cls_folder = root_dir / "Tomato___Early_blight"
    cls_folder.mkdir(parents=True)

    records = []
    for i in range(12):
        # Create patterned images
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
        arr[i * 10 : (i + 1) * 20, :, 0] = 180
        arr[:, i * 10 : (i + 1) * 20, 1] = 200
        img = Image.fromarray(arr)
        img_path = cls_folder / f"leaf_{i:03d}.jpg"
        img.save(img_path)

        records.append({
            "path": f"Tomato___Early_blight/leaf_{i:03d}.jpg",
            "label": "Tomato___Early_blight",
            "label_idx": 29,
        })

    df = pd.DataFrame(records)
    train_df = df.iloc[:8].copy().reset_index(drop=True)
    val_df = df.iloc[8:10].copy().reset_index(drop=True)
    test_df = df.iloc[10:].copy().reset_index(drop=True)

    return root_dir, train_df, val_df, test_df


def test_transform_shapes_and_types():
    dummy_img = Image.fromarray(np.ones((256, 256, 3), dtype=np.uint8) * 128)

    train_tf = get_train_transforms(img_size=224)
    train_out = train_tf(dummy_img)
    assert isinstance(train_out, torch.Tensor)
    assert train_out.shape == (3, 224, 224)
    assert train_out.dtype == torch.float32

    eval_tf = get_eval_transforms(img_size=224, resize_size=256)
    eval_out = eval_tf(dummy_img)
    assert isinstance(eval_out, torch.Tensor)
    assert eval_out.shape == (3, 224, 224)
    assert eval_out.dtype == torch.float32


def test_eval_transform_determinism():
    arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    img = Image.fromarray(arr)

    eval_tf = get_eval_transforms(img_size=224)
    out1 = eval_tf(img)
    out2 = eval_tf(img)

    assert torch.allclose(out1, out2), "Eval transform must be strictly deterministic!"


def test_train_transform_stochasticity():
    arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    img = Image.fromarray(arr)

    train_tf = get_train_transforms(img_size=224)
    out1 = train_tf(img)
    out2 = train_tf(img)

    assert not torch.allclose(out1, out2), "Train transform should exhibit stochastic augmentations."


def test_denormalize_tensor():
    dummy_img = Image.fromarray(np.full((256, 256, 3), 100, dtype=np.uint8))
    eval_tf = get_eval_transforms()
    normalized_t = eval_tf(dummy_img)

    denorm = denormalize_tensor(normalized_t)
    assert denorm.min() >= 0.0
    assert denorm.max() <= 1.0


def test_plant_dataset_and_dataloaders(mock_image_environment):
    root_dir, train_df, val_df, test_df = mock_image_environment

    dataset = PlantDataset(csv_source=train_df, root_dir=root_dir, transform=get_eval_transforms())
    assert len(dataset) == 8

    img_tensor, label_tensor = dataset[0]
    assert img_tensor.shape == (3, 224, 224)
    assert isinstance(label_tensor, torch.Tensor)
    assert label_tensor.item() == 29
    assert label_tensor.dtype == torch.long

    train_loader, val_loader, test_loader = get_dataloaders(
        train_manifest=train_df,
        val_manifest=val_df,
        test_manifest=test_df,
        root_dir=root_dir,
        batch_size=4,
        num_workers=0,
    )

    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.shape == (4, 3, 224, 224)
    assert batch_y.shape == (4,)
    assert batch_y.dtype == torch.long


def test_field_dataset(tmp_path: Path):
    field_root = tmp_path / "field_root"
    field_folder = field_root / "Tomato___Early_blight"
    field_folder.mkdir(parents=True)

    img = Image.fromarray(np.full((256, 256, 3), 150, dtype=np.uint8))
    img.save(field_folder / "test_field_001.jpg")

    field_df = pd.DataFrame([{
        "filepath": "Tomato___Early_blight/test_field_001.jpg",
        "filename": "test_field_001.jpg",
        "label": "Tomato___Early_blight",
        "label_idx": 29,
        "source": "PlantDoc",
    }])

    f_dataset = FieldDataset(csv_source=field_df, root_dir=field_root)
    img_t, target_t = f_dataset[0]
    assert img_t.shape == (3, 224, 224)
    assert target_t.item() == 29


def test_weighted_sampler(mock_image_environment):
    root_dir, train_df, _, _ = mock_image_environment
    dataset = PlantDataset(csv_source=train_df, root_dir=root_dir)

    mock_weights = [1.0] * 38
    mock_weights[29] = 2.5
    sampler = create_weighted_sampler(dataset, mock_weights)

    assert len(sampler) == len(dataset)
    indices = list(sampler)
    assert len(indices) == len(dataset)
