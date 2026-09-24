import os
import hashlib
import csv
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

A_TEST = Path(r"C:\user\juli\thesis\Dataset\orgnl_preprocessed_testt")

B_TRAIN = Path(r"C:\user\juli\thesis\Dataset\preprocessed_train")

B_TEST = Path(r"C:\user\juli\thesis\Dataset\preprocessed_test")


# ============================================================
# OUTPUT FILES
# ============================================================

OUTPUT_DIR = Path(r"C:\user\juli\thesis\Dataset\overlap_check_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TXT_OUTPUT = OUTPUT_DIR / "overlap_report.txt"
CSV_OUTPUT = OUTPUT_DIR / "overlap_pairs.csv"


# ============================================================
# HASH FUNCTION
# ============================================================

def file_hash(path, chunk_size=1024 * 1024):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


# ============================================================
# COLLECT HASHES
# ============================================================

def collect_hashes(folder):

    hashes = {}

    for path in folder.rglob("*"):

        if path.is_file():

            try:

                h = file_hash(path)

                hashes[h] = str(path)

            except Exception as e:

                print(f"Could not read: {path}")
                print(e)

    return hashes


# ============================================================
# START
# ============================================================

print("=" * 70)
print("CROSS-DATASET OVERLAP CHECK")
print("=" * 70)

print("\nScanning A-test...")
a_test_hashes = collect_hashes(A_TEST)
print(f"A-test files: {len(a_test_hashes)}")

print("\nScanning B-train...")
b_train_hashes = collect_hashes(B_TRAIN)
print(f"B-train files: {len(b_train_hashes)}")

print("\nScanning B-test...")
b_test_hashes = collect_hashes(B_TEST)
print(f"B-test files: {len(b_test_hashes)}")


# ============================================================
# FIND OVERLAPS
# ============================================================

overlap_train = set(a_test_hashes) & set(b_train_hashes)

overlap_test = set(a_test_hashes) & set(b_test_hashes)


# ============================================================
# WRITE TXT REPORT
# ============================================================

with open(TXT_OUTPUT, "w", encoding="utf-8") as report:

    report.write("=" * 70 + "\n")
    report.write("CROSS-DATASET OVERLAP CHECK REPORT\n")
    report.write("=" * 70 + "\n\n")

    report.write("DATASET COUNTS\n")
    report.write("-" * 70 + "\n")

    report.write(f"A-test files : {len(a_test_hashes)}\n")
    report.write(f"B-train files: {len(b_train_hashes)}\n")
    report.write(f"B-test files : {len(b_test_hashes)}\n\n")


    # --------------------------------------------------------
    # A-TEST vs B-TRAIN
    # --------------------------------------------------------

    report.write("=" * 70 + "\n")
    report.write("A-TEST vs B-TRAIN\n")
    report.write("=" * 70 + "\n")

    report.write(
        f"Exact duplicate files found: {len(overlap_train)}\n\n"
    )

    if overlap_train:

        for h in sorted(overlap_train):

            report.write("A-test : " + a_test_hashes[h] + "\n")
            report.write("B-train: " + b_train_hashes[h] + "\n")
            report.write("SHA256 : " + h + "\n")
            report.write("-" * 70 + "\n")

    else:

        report.write("No exact duplicate files found.\n")


    # --------------------------------------------------------
    # A-TEST vs B-TEST
    # --------------------------------------------------------

    report.write("\n")
    report.write("=" * 70 + "\n")
    report.write("A-TEST vs B-TEST\n")
    report.write("=" * 70 + "\n")

    report.write(
        f"Exact duplicate files found: {len(overlap_test)}\n\n"
    )

    if overlap_test:

        for h in sorted(overlap_test):

            report.write("A-test : " + a_test_hashes[h] + "\n")
            report.write("B-test : " + b_test_hashes[h] + "\n")
            report.write("SHA256 : " + h + "\n")
            report.write("-" * 70 + "\n")

    else:

        report.write("No exact duplicate files found.\n")


    # --------------------------------------------------------
    # CLEAN A-TEST
    # --------------------------------------------------------

    all_overlap = overlap_train | overlap_test

    clean_a_test_count = len(a_test_hashes) - len(all_overlap)

    report.write("\n")
    report.write("=" * 70 + "\n")
    report.write("A-TEST CLEANING SUMMARY\n")
    report.write("=" * 70 + "\n")

    report.write(f"Total A-test images       : {len(a_test_hashes)}\n")
    report.write(
        f"A-test overlapping B-train: {len(overlap_train)}\n"
    )
    report.write(
        f"A-test overlapping B-test : {len(overlap_test)}\n"
    )
    report.write(
        f"Unique overlapping images : {len(all_overlap)}\n"
    )
    report.write(
        f"Remaining non-overlap A-test: {clean_a_test_count}\n"
    )


    # --------------------------------------------------------
    # FINAL VERDICT
    # --------------------------------------------------------

    report.write("\n")
    report.write("=" * 70 + "\n")
    report.write("FINAL VERDICT\n")
    report.write("=" * 70 + "\n")

    if len(overlap_train) == 0:

        report.write("A-test does NOT overlap with B-train.\n")

    else:

        report.write(
            "A-test DOES overlap with B-train.\n"
        )

    if len(overlap_test) == 0:

        report.write("A-test does NOT overlap with B-test.\n")

    else:

        report.write(
            "A-test DOES overlap with B-test.\n"
        )

    report.write("\nDone.\n")


# ============================================================
# WRITE CSV FILE
# ============================================================

with open(
    CSV_OUTPUT,
    "w",
    newline="",
    encoding="utf-8"
) as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow([
        "A-test",
        "Other Dataset",
        "Other Path",
        "SHA256"
    ])

    for h in sorted(overlap_train):

        writer.writerow([
            a_test_hashes[h],
            "B-train",
            b_train_hashes[h],
            h
        ])

    for h in sorted(overlap_test):

        writer.writerow([
            a_test_hashes[h],
            "B-test",
            b_test_hashes[h],
            h
        ])


# ============================================================
# FINAL TERMINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(f"A-test total              : {len(a_test_hashes)}")
print(f"B-train total             : {len(b_train_hashes)}")
print(f"B-test total              : {len(b_test_hashes)}")

print(f"\nA-test vs B-train overlap : {len(overlap_train)}")
print(f"A-test vs B-test overlap  : {len(overlap_test)}")

print(f"\nUnique overlapping A-test : {len(all_overlap)}")
print(f"Clean A-test remaining    : {clean_a_test_count}")

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(f"\nText report:")
print(TXT_OUTPUT)

print(f"\nCSV report:")
print(CSV_OUTPUT)

print("\nDone.")