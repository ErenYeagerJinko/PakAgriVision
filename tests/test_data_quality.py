"""Unit tests for dataset quality inspection, corruption detection, and deduplication."""

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from data_scripts.check_dataset import write_report_csv
from src.data.quality import QualityChecker, compute_md5, compute_phash, verify_image


@pytest.fixture
def test_dataset_dir(tmp_path: Path) -> Path:
    """Create a temporary dataset structure with synthetic images."""
    data_dir = tmp_path / "mock_dataset"
    class_a = data_dir / "Tomato___Early_blight"
    class_b = data_dir / "Tomato___healthy"
    class_a.mkdir(parents=True)
    class_b.mkdir(parents=True)

    # 1. Clean valid image
    img1 = Image.fromarray(np.full((224, 224, 3), 120, dtype=np.uint8))
    img1_path = class_a / "clean_001.jpg"
    img1.save(img1_path)

    # 2. Exact duplicate of clean_001
    img1_copy_path = class_a / "clean_001_exact_copy.jpg"
    img1_copy_path.write_bytes(img1_path.read_bytes())

    # 3. Near-duplicate (identical content with minor 1-pixel perturbation)
    arr_near = np.full((224, 224, 3), 120, dtype=np.uint8)
    arr_near[0:2, 0:2] = 125
    img_near = Image.fromarray(arr_near)
    img_near.save(class_a / "clean_001_near_copy.jpg")

    # 4. Distinct image in class A
    arr_distinct = np.zeros((224, 224, 3), dtype=np.uint8)
    arr_distinct[50:150, 50:150] = 255
    Image.fromarray(arr_distinct).save(class_a / "distinct_002.jpg")

    # 5. Corrupt image file (garbage bytes)
    corrupt_path = class_a / "corrupt_003.jpg"
    corrupt_path.write_bytes(b"This is definitely not a valid JPEG image file.")

    # 6. Cross-class duplicate: clean_001 placed into class_b
    cross_class_path = class_b / "cross_class_copy.jpg"
    cross_class_path.write_bytes(img1_path.read_bytes())

    return data_dir


def test_verify_image_valid(test_dataset_dir: Path):
    valid_file = test_dataset_dir / "Tomato___Early_blight" / "clean_001.jpg"
    is_valid, err, w, h, channels = verify_image(valid_file)
    assert is_valid is True
    assert err is None
    assert (w, h) == (224, 224)
    assert channels == 3


def test_verify_image_corrupt(test_dataset_dir: Path):
    corrupt_file = test_dataset_dir / "Tomato___Early_blight" / "corrupt_003.jpg"
    is_valid, err, w, h, channels = verify_image(corrupt_file)
    assert is_valid is False
    assert err is not None


def test_hashing_functions(test_dataset_dir: Path):
    f1 = test_dataset_dir / "Tomato___Early_blight" / "clean_001.jpg"
    f2 = test_dataset_dir / "Tomato___Early_blight" / "clean_001_exact_copy.jpg"

    md5_1 = compute_md5(f1)
    md5_2 = compute_md5(f2)
    assert md5_1 == md5_2
    assert len(md5_1) == 32

    phash_1 = compute_phash(f1)
    phash_2 = compute_phash(f2)
    assert phash_1 == phash_2


def test_deduplication_pipeline(test_dataset_dir: Path):
    checker = QualityChecker(phash_threshold=4)
    records = []

    for file_path in test_dataset_dir.rglob("*.jpg"):
        rel_path = file_path.relative_to(test_dataset_dir).as_posix()
        label = file_path.parent.name
        records.append(checker.inspect_file(file_path, rel_path, label))

    results = checker.run_deduplication(records)
    results_by_path = {r.path: r for r in results}

    # Verify corrupt
    assert results_by_path["Tomato___Early_blight/corrupt_003.jpg"].hash_type == "corrupt"
    assert results_by_path["Tomato___Early_blight/corrupt_003.jpg"].is_valid is False

    # Verify distinct
    assert results_by_path["Tomato___Early_blight/distinct_002.jpg"].hash_type == "unique"

    # Verify cross-class duplicate is detected
    cross_copy = results_by_path["Tomato___healthy/cross_class_copy.jpg"]
    assert cross_copy.hash_type == "cross_class_duplicate"


def test_report_csv_generation(tmp_path: Path, test_dataset_dir: Path):
    checker = QualityChecker(phash_threshold=4)
    records = []
    for file_path in test_dataset_dir.rglob("*.jpg"):
        rel_path = file_path.relative_to(test_dataset_dir).as_posix()
        label = file_path.parent.name
        records.append(checker.inspect_file(file_path, rel_path, label))

    results = checker.run_deduplication(records)
    csv_path = tmp_path / "dedupe_report.csv"
    write_report_csv(results, csv_path)

    assert csv_path.exists()
    with csv_path.open(encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == len(results)
        expected_cols = {"group_id", "path", "label", "hash_type", "width", "height", "md5", "phash", "notes"}
        assert expected_cols.issubset(set(reader[0].keys()))
