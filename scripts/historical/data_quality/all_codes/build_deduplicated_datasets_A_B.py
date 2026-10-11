"""Build leakage-safe, exactly deduplicated Dataset A and Dataset B partitions.

This is a standalone script. It NEVER modifies or deletes source images.

Policy
------
1. Images are compared by SHA-256 of normalized grayscale pixels (256 x 256).
2. Duplicate copies inside Dataset A are reduced to one representative.
3. Every image occurring in Dataset A is excluded from Dataset B. Dataset A is
   therefore the primary dataset and Dataset B remains exactly independent.
4. Remaining unique images in each dataset are repartitioned separately using
   a deterministic, class-stratified 80/20 train/test split.
5. New images, manifests, and an audit summary are written to a new folder.

Near-duplicate perceptual hashes are intentionally not removed automatically:
similar MRI slices can be false positives and require visual/patient-ID review.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError
from tqdm.auto import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Record:
    dataset: str
    old_split: str
    class_name: str
    source_path: Path
    pixel_hash: str
    width: int
    height: int
    file_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create new, exactly deduplicated A/B train-test partitions."
    )
    parser.add_argument(
        "--a-train",
        type=Path,
        default=Path(r"C:\user\juli\thesis\original_file\othr folders\preprocessed_trainn"),
    )
    parser.add_argument(
        "--a-test",
        type=Path,
        default=Path(r"C:\user\juli\thesis\original_file\othr folders\preprocessed_testt"),
    )
    parser.add_argument(
        "--b-train",
        type=Path,
        default=Path(r"C:\user\juli\thesis\Dataset_B\Dataset\preprocessed_train"),
    )
    parser.add_argument(
        "--b-test",
        type=Path,
        default=Path(r"C:\user\juli\thesis\Dataset_B\Dataset\preprocessed_test"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            r"C:\user\juli\thesis\original_file\othr folders\clean_deduplicated_datasets"
        ),
    )
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def discover(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Source folder does not exist: {root}")
    return sorted(
        p.resolve()
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def normalized_pixel_hash(image: Image.Image) -> str:
    normalized = ImageOps.exif_transpose(image).convert("L").resize(
        (256, 256), Image.Resampling.BILINEAR
    )
    return hashlib.sha256(normalized.tobytes()).hexdigest()


def scan(dataset: str, old_split: str, root: Path) -> tuple[list[Record], list[dict]]:
    records: list[Record] = []
    errors: list[dict] = []
    paths = discover(root)
    for path in tqdm(paths, desc=f"Hashing {dataset}_{old_split}", unit="image"):
        try:
            relative = path.relative_to(root.resolve())
            class_name = relative.parts[0] if len(relative.parts) > 1 else path.parent.name
            with Image.open(path) as image:
                width, height = image.size
                digest = normalized_pixel_hash(image)
            records.append(
                Record(
                    dataset=dataset,
                    old_split=old_split,
                    class_name=class_name,
                    source_path=path,
                    pixel_hash=digest,
                    width=width,
                    height=height,
                    file_size=path.stat().st_size,
                )
            )
        except (OSError, ValueError, UnidentifiedImageError) as error:
            errors.append(
                {
                    "dataset": dataset,
                    "old_split": old_split,
                    "path": str(path),
                    "error": repr(error),
                }
            )
    return records, errors


def choose_representative(group: Iterable[Record]) -> Record:
    # Prefer the largest original resolution, then file size, then stable path.
    return sorted(
        group,
        key=lambda r: (-(r.width * r.height), -r.file_size, str(r.source_path).casefold()),
    )[0]


def deduplicate_dataset(records: list[Record]) -> tuple[list[Record], list[dict]]:
    groups: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        groups[record.pixel_hash].append(record)

    kept: list[Record] = []
    decisions: list[dict] = []
    for digest in sorted(groups):
        group = groups[digest]
        classes = {r.class_name.casefold() for r in group}
        if len(classes) != 1:
            paths = "\n".join(str(r.source_path) for r in group)
            raise RuntimeError(
                "Identical pixels have conflicting class labels. Resolve these before "
                f"continuing:\n{paths}"
            )
        representative = choose_representative(group)
        kept.append(representative)
        for record in group:
            decisions.append(
                {
                    "dataset": record.dataset,
                    "pixel_hash": digest,
                    "class_name": record.class_name,
                    "old_split": record.old_split,
                    "source_path": str(record.source_path),
                    "representative_path": str(representative.source_path),
                    "decision": "keep_representative"
                    if record == representative
                    else "remove_internal_exact_duplicate",
                }
            )
    return kept, decisions


def stratified_split(
    records: list[Record], test_fraction: float, seed: int
) -> tuple[list[Record], list[Record]]:
    by_class: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        by_class[record.class_name].append(record)

    train: list[Record] = []
    test: list[Record] = []
    for class_index, class_name in enumerate(sorted(by_class, key=str.casefold)):
        items = sorted(by_class[class_name], key=lambda r: r.pixel_hash)
        random.Random(seed + class_index * 1009).shuffle(items)
        if len(items) < 2:
            raise RuntimeError(f"Class {class_name!r} has fewer than two unique images.")
        test_count = max(1, min(len(items) - 1, int(round(len(items) * test_fraction))))
        test.extend(items[:test_count])
        train.extend(items[test_count:])
    return train, test


def safe_filename(record: Record) -> str:
    suffix = record.source_path.suffix.lower()
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in record.source_path.stem)
    return f"{record.pixel_hash[:16]}_{stem}{suffix}"


def copy_partition(
    records: list[Record], dataset: str, split: str, output_dir: Path
) -> list[dict]:
    rows: list[dict] = []
    for record in tqdm(records, desc=f"Copying {dataset}_{split}", unit="image"):
        destination = output_dir / f"Dataset_{dataset}" / split / record.class_name / safe_filename(record)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(
                f"Output file already exists: {destination}\n"
                "Use an empty output directory to avoid mixing runs."
            )
        shutil.copy2(record.source_path, destination)
        rows.append(
            {
                "dataset": dataset,
                "new_split": split,
                "class_name": record.class_name,
                "pixel_hash": record.pixel_hash,
                "source_path": str(record.source_path),
                "destination_path": str(destination.resolve()),
                "old_split": record.old_split,
            }
        )
    return rows


def counts(records: Iterable[Record]) -> Counter:
    return Counter(r.class_name for r in records)


def verify_no_overlap(partitions: dict[str, list[Record]]) -> None:
    names = list(partitions)
    for i, left_name in enumerate(names):
        left_hashes = {r.pixel_hash for r in partitions[left_name]}
        if len(left_hashes) != len(partitions[left_name]):
            raise RuntimeError(f"Internal duplicates remain in {left_name}.")
        for right_name in names[i + 1 :]:
            right_hashes = {r.pixel_hash for r in partitions[right_name]}
            overlap = left_hashes & right_hashes
            if overlap:
                raise RuntimeError(
                    f"Verification failed: {len(overlap)} exact hashes overlap between "
                    f"{left_name} and {right_name}."
                )


def main() -> None:
    args = parse_args()
    if not 0.0 < args.test_fraction < 1.0:
        raise ValueError("--test-fraction must be between 0 and 1.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}\n"
            "Rename/remove the previous output or choose another --output-dir."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    roots = {
        ("A", "train"): args.a_train,
        ("A", "test"): args.a_test,
        ("B", "train"): args.b_train,
        ("B", "test"): args.b_test,
    }
    all_records: list[Record] = []
    errors: list[dict] = []
    for (dataset, split), root in roots.items():
        scanned, failures = scan(dataset, split, root)
        all_records.extend(scanned)
        errors.extend(failures)
    pd.DataFrame(errors, columns=["dataset", "old_split", "path", "error"]).to_csv(
        args.output_dir / "unreadable_images.csv", index=False
    )
    if errors:
        raise RuntimeError(
            f"Found {len(errors)} unreadable images. See unreadable_images.csv; no clean "
            "dataset was created."
        )

    a_original = [r for r in all_records if r.dataset == "A"]
    b_original = [r for r in all_records if r.dataset == "B"]
    a_unique, a_decisions = deduplicate_dataset(a_original)
    b_unique_before_cross, b_decisions = deduplicate_dataset(b_original)

    a_hashes = {r.pixel_hash for r in a_unique}
    b_unique = [r for r in b_unique_before_cross if r.pixel_hash not in a_hashes]
    removed_b_cross = [r for r in b_unique_before_cross if r.pixel_hash in a_hashes]
    for record in removed_b_cross:
        b_decisions.append(
            {
                "dataset": "B",
                "pixel_hash": record.pixel_hash,
                "class_name": record.class_name,
                "old_split": record.old_split,
                "source_path": str(record.source_path),
                "representative_path": "Dataset A",
                "decision": "remove_exact_overlap_with_dataset_A",
            }
        )

    a_train, a_test = stratified_split(a_unique, args.test_fraction, args.seed)
    b_train, b_test = stratified_split(b_unique, args.test_fraction, args.seed)
    partitions = {
        "A_train": a_train,
        "A_test": a_test,
        "B_train": b_train,
        "B_test": b_test,
    }
    verify_no_overlap(partitions)

    manifest_rows: list[dict] = []
    for name, records in partitions.items():
        dataset, split = name.split("_")
        manifest_rows.extend(copy_partition(records, dataset, split, args.output_dir))

    pd.DataFrame(manifest_rows).to_csv(args.output_dir / "clean_manifest.csv", index=False)
    pd.DataFrame(a_decisions + b_decisions).to_csv(
        args.output_dir / "deduplication_decisions.csv", index=False
    )

    summary_rows = []
    for name, records in partitions.items():
        dataset, split = name.split("_")
        per_class = counts(records)
        for class_name in sorted(per_class, key=str.casefold):
            summary_rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "class_name": class_name,
                    "image_count": per_class[class_name],
                }
            )
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "partition_summary.csv", index=False)

    report = [
        "EXACT DEDUPLICATION REPORT",
        "==========================",
        f"Seed: {args.seed}",
        f"Test fraction: {args.test_fraction:.4f}",
        f"Dataset A original images: {len(a_original)}",
        f"Dataset A unique images retained: {len(a_unique)}",
        f"Dataset A internal duplicate copies removed: {len(a_original) - len(a_unique)}",
        f"Dataset B original images: {len(b_original)}",
        f"Dataset B unique before cross-dataset removal: {len(b_unique_before_cross)}",
        f"Dataset B internal duplicate copies removed: {len(b_original) - len(b_unique_before_cross)}",
        f"Dataset B unique images removed because present in A: {len(removed_b_cross)}",
        f"Dataset B independent unique images retained: {len(b_unique)}",
        "",
        "NEW PARTITIONS",
    ]
    for name, records in partitions.items():
        report.append(f"{name}: {len(records)}; classes={dict(sorted(counts(records).items()))}")
    report.extend(
        [
            "",
            "Verification: PASS - zero exact normalized-pixel hashes overlap among any new partitions.",
            "Important: patient-level independence cannot be guaranteed without patient IDs.",
            "Perceptual near-duplicates were not automatically deleted and require manual review.",
        ]
    )
    (args.output_dir / "DEDUPLICATION_REPORT.txt").write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))
    print(f"\nClean datasets saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
