"""Read-only duplicate and overlap audit for brain MRI Datasets A and B.

The script never changes source images. It checks four partitions:
Dataset A train/test and Dataset B train/test. Three duplicate definitions are
reported:

1. Exact file: identical SHA-256 bytes.
2. Exact pixels: identical grayscale pixels after resizing to 256 x 256.
3. Near duplicate: 64-bit difference hash within a configurable Hamming distance.

Run this file directly. Results are written as CSV and TXT files.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError
from tqdm.auto import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PARTITION_ORDER = ["A_train", "A_test", "B_train", "B_test"]


@dataclass(frozen=True)
class ImageRecord:
    partition: str
    dataset: str
    split: str
    class_name: str
    path: str
    file_size: int
    sha256: str
    pixel_sha256: str
    dhash_hex: str


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def normalized_pixels_and_dhash(path: Path) -> Tuple[str, str]:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L")
        normalized = image.resize((256, 256), Image.Resampling.BILINEAR)
        pixel_digest = hashlib.sha256(np.asarray(normalized, dtype=np.uint8).tobytes()).hexdigest()

        small = image.resize((9, 8), Image.Resampling.LANCZOS)
        pixels = np.asarray(small, dtype=np.int16)
        differences = pixels[:, 1:] > pixels[:, :-1]
        value = 0
        for bit in differences.ravel():
            value = (value << 1) | int(bit)
        return pixel_digest, f"{value:016x}"


def discover_images(root: Path) -> List[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset partition does not exist: {root}")
    return sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def scan_partition(partition: str, root: Path) -> Tuple[List[ImageRecord], List[Dict[str, str]]]:
    dataset, split = partition.split("_", maxsplit=1)
    records: List[ImageRecord] = []
    failures: List[Dict[str, str]] = []
    paths = discover_images(root)
    for path in tqdm(paths, desc=f"Hashing {partition}", unit="image", dynamic_ncols=True):
        try:
            pixel_hash, dhash = normalized_pixels_and_dhash(path)
            relative = path.relative_to(root.resolve())
            class_name = relative.parts[0] if len(relative.parts) > 1 else path.parent.name
            records.append(
                ImageRecord(
                    partition=partition,
                    dataset=dataset,
                    split=split,
                    class_name=class_name,
                    path=str(path),
                    file_size=path.stat().st_size,
                    sha256=file_sha256(path),
                    pixel_sha256=pixel_hash,
                    dhash_hex=dhash,
                )
            )
        except (OSError, ValueError, UnidentifiedImageError) as error:
            failures.append({"partition": partition, "path": str(path), "error": repr(error)})
    return records, failures


def pair_row(left: ImageRecord, right: ImageRecord, match_type: str, distance: int = 0) -> Dict:
    return {
        "match_type": match_type,
        "hamming_distance": distance,
        "left_partition": left.partition,
        "left_class": left.class_name,
        "left_path": left.path,
        "right_partition": right.partition,
        "right_class": right.class_name,
        "right_path": right.path,
        "same_class": left.class_name.casefold() == right.class_name.casefold(),
    }


def cross_partition_exact_pairs(
    records: Sequence[ImageRecord], attribute: str, match_type: str
) -> List[Dict]:
    grouped: Dict[str, List[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[getattr(record, attribute)].append(record)
    rows: List[Dict] = []
    for members in grouped.values():
        partitions = {member.partition for member in members}
        if len(partitions) < 2:
            continue
        for left, right in itertools.combinations(members, 2):
            if left.partition != right.partition:
                rows.append(pair_row(left, right, match_type))
    return rows


def within_partition_duplicate_groups(
    records: Sequence[ImageRecord], attribute: str, match_type: str
) -> List[Dict]:
    grouped: Dict[Tuple[str, str], List[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.partition, getattr(record, attribute))].append(record)
    rows: List[Dict] = []
    group_number = 0
    for (partition, digest), members in grouped.items():
        if len(members) < 2:
            continue
        group_number += 1
        for member in members:
            rows.append(
                {
                    "duplicate_type": match_type,
                    "group": group_number,
                    "partition": partition,
                    "digest": digest,
                    "class_name": member.class_name,
                    "path": member.path,
                    "group_size": len(members),
                }
            )
    return rows


class BKNode:
    def __init__(self, value: int, indices: Optional[List[int]] = None):
        self.value = value
        self.indices = indices or []
        self.children: Dict[int, "BKNode"] = {}


class HammingBKTree:
    def __init__(self):
        self.root: Optional[BKNode] = None

    @staticmethod
    def distance(left: int, right: int) -> int:
        return (left ^ right).bit_count()

    def add(self, value: int, index: int) -> None:
        if self.root is None:
            self.root = BKNode(value, [index])
            return
        node = self.root
        while True:
            distance = self.distance(value, node.value)
            if distance == 0:
                node.indices.append(index)
                return
            if distance not in node.children:
                node.children[distance] = BKNode(value, [index])
                return
            node = node.children[distance]

    def query(self, value: int, threshold: int) -> Iterable[Tuple[int, int]]:
        if self.root is None:
            return
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = self.distance(value, node.value)
            if distance <= threshold:
                for index in node.indices:
                    yield index, distance
            lower = distance - threshold
            upper = distance + threshold
            for edge, child in node.children.items():
                if lower <= edge <= upper:
                    stack.append(child)


def near_duplicate_pairs(
    records: Sequence[ImageRecord], threshold: int, exact_pixel_pairs: Sequence[Dict]
) -> List[Dict]:
    by_partition = {
        partition: [record for record in records if record.partition == partition]
        for partition in PARTITION_ORDER
    }
    exact_paths = {
        frozenset((row["left_path"], row["right_path"])) for row in exact_pixel_pairs
    }
    rows: List[Dict] = []
    comparisons = list(itertools.combinations(PARTITION_ORDER, 2))
    for left_partition, right_partition in comparisons:
        left_records = by_partition[left_partition]
        right_records = by_partition[right_partition]
        tree = HammingBKTree()
        for index, record in enumerate(left_records):
            tree.add(int(record.dhash_hex, 16), index)
        for right in tqdm(
            right_records,
            desc=f"Near {left_partition} vs {right_partition}",
            unit="image",
            dynamic_ncols=True,
        ):
            for index, distance in tree.query(int(right.dhash_hex, 16), threshold):
                left = left_records[index]
                if frozenset((left.path, right.path)) in exact_paths:
                    continue
                rows.append(pair_row(left, right, "near_dhash", distance))
    return rows


def comparison_summary(pair_frame: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for left_partition, right_partition in itertools.combinations(PARTITION_ORDER, 2):
        selected = pair_frame[
            ((pair_frame["left_partition"] == left_partition) &
             (pair_frame["right_partition"] == right_partition))
            | ((pair_frame["left_partition"] == right_partition) &
               (pair_frame["right_partition"] == left_partition))
        ]
        left_paths = set(selected["left_path"]) | set(selected["right_path"])
        left_root_paths = set(inventory.loc[inventory["partition"] == left_partition, "path"])
        right_root_paths = set(inventory.loc[inventory["partition"] == right_partition, "path"])
        affected_left = len(left_paths & left_root_paths)
        affected_right = len(left_paths & right_root_paths)
        rows.append(
            {
                "left_partition": left_partition,
                "right_partition": right_partition,
                "matching_pairs": len(selected),
                "affected_left_images": affected_left,
                "left_total_images": len(left_root_paths),
                "affected_left_percent": 100 * affected_left / len(left_root_paths) if left_root_paths else 0,
                "affected_right_images": affected_right,
                "right_total_images": len(right_root_paths),
                "affected_right_percent": 100 * affected_right / len(right_root_paths) if right_root_paths else 0,
            }
        )
    return pd.DataFrame(rows)


def write_text_report(
    path: Path,
    roots: Dict[str, Path],
    inventory: pd.DataFrame,
    exact_file: pd.DataFrame,
    exact_pixel: pd.DataFrame,
    near: pd.DataFrame,
    summary: pd.DataFrame,
    failures: pd.DataFrame,
    near_threshold: int,
) -> None:
    lines = [
        "DATASET A AND B OVERLAP AUDIT",
        "=" * 78,
        "This report is descriptive. Source images were not modified.",
        "",
        "PARTITIONS",
        "-" * 78,
    ]
    for partition in PARTITION_ORDER:
        count = int((inventory["partition"] == partition).sum())
        lines.append(f"{partition:8s}: {count:6d} images | {roots[partition]}")
    lines.extend(
        [
            "",
            "CROSS-PARTITION RESULTS",
            "-" * 78,
            f"Exact byte-identical pairs     : {len(exact_file)}",
            f"Exact normalized-pixel pairs  : {len(exact_pixel)}",
            f"Near-duplicate dHash pairs    : {len(near)} (distance <= {near_threshold})",
            f"Unreadable images             : {len(failures)}",
            "",
            summary.to_string(index=False),
            "",
            "INTERPRETATION",
            "-" * 78,
            "Any train-test overlap can inflate evaluation performance.",
            "Exact-file and exact-pixel matches are strong duplicate evidence.",
            "Near-dHash matches are candidates and must be visually reviewed before removal.",
            "Same-class matches can still cause leakage; different-class matches also require label review.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--a-train",
        default=r"C:\user\juli\thesis\original_file\othr folders\preprocessed_trainn",
    )
    parser.add_argument(
        "--a-test",
        default=r"C:\user\juli\thesis\original_file\othr folders\preprocessed_testt",
    )
    parser.add_argument(
        "--b-train",
        default=r"C:\user\juli\thesis\Dataset_B\Dataset\preprocessed_train",
    )
    parser.add_argument(
        "--b-test",
        default=r"C:\user\juli\thesis\Dataset_B\Dataset\preprocessed_test",
    )
    parser.add_argument(
        "--output-dir",
        default=r"C:\user\juli\thesis\original_file\othr folders\dataset_overlap_A_B_results",
    )
    parser.add_argument(
        "--near-threshold",
        type=int,
        default=4,
        help="Maximum 64-bit dHash Hamming distance; 0 disables near-duplicate search.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.near_threshold <= 10:
        raise ValueError("--near-threshold must be between 0 and 10.")
    roots = {
        "A_train": Path(args.a_train).resolve(),
        "A_test": Path(args.a_test).resolve(),
        "B_train": Path(args.b_train).resolve(),
        "B_test": Path(args.b_test).resolve(),
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records: List[ImageRecord] = []
    failures: List[Dict[str, str]] = []
    for partition in PARTITION_ORDER:
        partition_records, partition_failures = scan_partition(partition, roots[partition])
        records.extend(partition_records)
        failures.extend(partition_failures)

    inventory = pd.DataFrame([record.__dict__ for record in records])
    failure_frame = pd.DataFrame(failures, columns=["partition", "path", "error"])
    exact_file_rows = cross_partition_exact_pairs(records, "sha256", "exact_file")
    exact_pixel_rows = cross_partition_exact_pairs(records, "pixel_sha256", "exact_pixels_256")
    exact_file = pd.DataFrame(exact_file_rows, columns=list(pair_row(records[0], records[0], "x")))
    exact_pixel = pd.DataFrame(exact_pixel_rows, columns=list(pair_row(records[0], records[0], "x")))
    near_rows = (
        near_duplicate_pairs(records, args.near_threshold, exact_pixel_rows)
        if args.near_threshold > 0
        else []
    )
    near = pd.DataFrame(near_rows, columns=list(pair_row(records[0], records[0], "x")))
    all_cross_pairs = pd.concat([exact_file, exact_pixel, near], ignore_index=True)
    summary = comparison_summary(all_cross_pairs, inventory)
    within_rows = within_partition_duplicate_groups(records, "sha256", "exact_file")
    within_rows += within_partition_duplicate_groups(records, "pixel_sha256", "exact_pixels_256")
    within = pd.DataFrame(within_rows)

    inventory.to_csv(output_dir / "image_inventory_and_hashes.csv", index=False)
    exact_file.to_csv(output_dir / "cross_partition_exact_file_pairs.csv", index=False)
    exact_pixel.to_csv(output_dir / "cross_partition_exact_pixel_pairs.csv", index=False)
    near.to_csv(output_dir / "cross_partition_near_duplicate_candidates.csv", index=False)
    within.to_csv(output_dir / "within_partition_duplicate_groups.csv", index=False)
    summary.to_csv(output_dir / "overlap_summary.csv", index=False)
    failure_frame.to_csv(output_dir / "unreadable_images.csv", index=False)
    write_text_report(
        output_dir / "overlap_report.txt",
        roots,
        inventory,
        exact_file,
        exact_pixel,
        near,
        summary,
        failure_frame,
        args.near_threshold,
    )
    print(f"Audit complete. Results saved to: {output_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
