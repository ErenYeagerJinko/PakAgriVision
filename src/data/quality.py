"""Data quality verification, corruption detection, and perceptual deduplication.

This module provides tools to:
1. Detect corrupt, unreadable, or truncated image files via Pillow.
2. Identify exact duplicates using MD5 byte-level hashing.
3. Identify near-duplicates using perceptual hashing (pHash) with a Hamming distance threshold.
4. Flag cross-class duplicates (identical/near-identical images assigned to multiple classes).
5. Cluster duplicate images into discrete group IDs via Union-Find to prevent leakage across splits.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import imagehash
from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ImageRecord:
    """Record describing a scanned dataset image."""

    path: str
    absolute_path: Path
    label: str
    is_valid: bool = False
    error: Optional[str] = None
    width: int = 0
    height: int = 0
    channels: int = 0
    md5: Optional[str] = None
    phash: Optional[str] = None
    group_id: Optional[str] = None
    hash_type: str = "unique"
    notes: str = ""


class UnionFind:
    """Disjoint-set data structure to cluster duplicate items into connected groups."""

    def __init__(self, items: Iterable[str]):
        self.parent: Dict[str, str] = {item: item for item in items}
        self.rank: Dict[str, int] = {item: 0 for item in items}

    def find(self, item: str) -> str:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1


def scan_image_files(root_dir: Path | str) -> List[Tuple[Path, str, str]]:
    """Scan root_dir recursively for supported image files.

    Returns:
        List of tuples: (absolute_path, relative_path_str, class_label).
    """
    root_path = Path(root_dir).resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root directory does not exist: {root_path}")

    found: List[Tuple[Path, str, str]] = []
    for file_path in root_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            relative_str = file_path.relative_to(root_path).as_posix()
            label = file_path.parent.name
            found.append((file_path, relative_str, label))

    found.sort(key=lambda x: x[1])
    return found


def verify_image(file_path: Path | str) -> Tuple[bool, Optional[str], int, int, int]:
    """Verify image binary integrity and RGB consistency using Pillow.

    Returns:
        (is_valid, error_message, width, height, channels)
    """
    p = Path(file_path)
    try:
        # Step 1: verify file headers & structural integrity
        with Image.open(p) as img:
            img.verify()

        # Step 2: re-open to inspect dimensions and convert to RGB
        with Image.open(p) as img:
            rgb_img = img.convert("RGB")
            width, height = rgb_img.size
            channels = len(rgb_img.getbands())

            if width <= 0 or height <= 0:
                return False, f"Invalid dimensions: {width}x{height}", 0, 0, 0

            return True, None, width, height, channels
    except Exception as exc:  # pylint: disable=broad-except
        return False, str(exc), 0, 0, 0


def compute_md5(file_path: Path | str, chunk_size: int = 65536) -> str:
    """Compute MD5 byte digest of a file."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_phash(file_path: Path | str, hash_size: int = 8) -> str:
    """Compute perceptual hash (pHash) string of an image."""
    with Image.open(file_path) as img:
        rgb_img = img.convert("RGB")
        h = imagehash.phash(rgb_img, hash_size=hash_size)
        return str(h)


class QualityChecker:
    """Orchestrates image verification, duplicate clustering, and report generation."""

    def __init__(self, phash_threshold: int = 4):
        """
        Args:
            phash_threshold: Maximum Hamming distance to consider two images near-duplicates.
        """
        self.phash_threshold = phash_threshold
        self.records: List[ImageRecord] = []

    def inspect_file(self, abs_path: Path, rel_path: str, label: str) -> ImageRecord:
        """Inspect a single file for corruption and calculate hashes."""
        is_valid, error, w, h, channels = verify_image(abs_path)
        record = ImageRecord(
            path=rel_path,
            absolute_path=abs_path,
            label=label,
            is_valid=is_valid,
            error=error,
            width=w,
            height=h,
            channels=channels,
        )

        if is_valid:
            try:
                record.md5 = compute_md5(abs_path)
                record.phash = compute_phash(abs_path)
            except Exception as exc:  # pylint: disable=broad-except
                record.is_valid = False
                record.error = f"Hashing failed: {exc}"

        return record

    def run_deduplication(self, records: List[ImageRecord]) -> List[ImageRecord]:
        """Cluster exact, near, and cross-class duplicates."""
        valid_records = [r for r in records if r.is_valid and r.md5 and r.phash]
        corrupt_records = [r for r in records if not r.is_valid]

        for r in corrupt_records:
            r.hash_type = "corrupt"
            r.group_id = f"corrupt_{r.path}"
            r.notes = r.error or "Corrupt file"

        if not valid_records:
            return records

        # 1. Exact duplicates via MD5
        md5_map: Dict[str, List[ImageRecord]] = {}
        for r in valid_records:
            md5_map.setdefault(r.md5, []).append(r)

        # 2. Setup Union-Find over all valid image paths
        uf = UnionFind([r.path for r in valid_records])

        # Link exact duplicates
        for md5_hash, group in md5_map.items():
            if len(group) > 1:
                first = group[0].path
                for other in group[1:]:
                    uf.union(first, other.path)

        # 3. Near-duplicates via pHash within each class
        # Convert hex pHash string back to imagehash.ImageHash for fast hamming distance
        phash_objects: Dict[str, imagehash.ImageHash] = {
            r.path: imagehash.hex_to_hash(r.phash) for r in valid_records
        }

        records_by_class: Dict[str, List[ImageRecord]] = {}
        for r in valid_records:
            records_by_class.setdefault(r.label, []).append(r)

        for label, class_records in records_by_class.items():
            n = len(class_records)
            for i in range(n):
                r1 = class_records[i]
                h1 = phash_objects[r1.path]
                for j in range(i + 1, n):
                    r2 = class_records[j]
                    h2 = phash_objects[r2.path]
                    distance = h1 - h2
                    if distance <= self.phash_threshold:
                        uf.union(r1.path, r2.path)

        # 4. Cross-class duplicates check
        # Check if identical MD5 or very close pHash spans across differing labels
        path_to_record = {r.path: r for r in valid_records}
        clusters: Dict[str, List[str]] = {}
        for r in valid_records:
            root = uf.find(r.path)
            clusters.setdefault(root, []).append(r.path)

        # Assign group_id and classify hash_type
        group_counter = 1
        for root_path, member_paths in clusters.items():
            labels_in_group = {path_to_record[p].label for p in member_paths}
            is_cross_class = len(labels_in_group) > 1
            has_duplicates = len(member_paths) > 1

            if has_duplicates:
                assigned_group_id = f"grp_{group_counter:05d}"
                group_counter += 1
            else:
                assigned_group_id = f"single_{member_paths[0]}"

            for p in member_paths:
                rec = path_to_record[p]
                rec.group_id = assigned_group_id

                if is_cross_class:
                    rec.hash_type = "cross_class_duplicate"
                    rec.notes = f"Labels involved: {sorted(labels_in_group)}"
                elif has_duplicates:
                    # check if exact or near
                    matching_md5 = len(md5_map[rec.md5]) > 1
                    rec.hash_type = "exact_duplicate" if matching_md5 else "near_duplicate"
                    rec.notes = f"Cluster size: {len(member_paths)}"
                else:
                    rec.hash_type = "unique"
                    rec.notes = "Unique clean image"

        return records
