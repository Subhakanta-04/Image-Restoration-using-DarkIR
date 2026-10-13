import argparse
import os
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = PROJECT_ROOT / "data" / "datasets" / "LOLBlur" / "train"
parser = argparse.ArgumentParser(description="Check LOL-Blur class/target filename and dimension alignment")
parser.add_argument(
    "--root",
    type=Path,
    default=Path(os.environ.get("DARKIR_LOLBLUR_TRAIN", DEFAULT_ROOT)),
    help="LOL-Blur train directory containing the condition folders",
)
ROOT = parser.parse_args().root

conditions = [
    "low_blur",
    "low_blur_noise",
    "low_sharp",
]

target = "high_sharp_scaled"

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect(folder):
    files = {}
    duplicates = []

    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in extensions:
            key = p.relative_to(folder).as_posix()

            if key in files:
                duplicates.append(key)
            else:
                files[key] = p

    return files, duplicates


print("=" * 70)
print("LOL-Blur Dataset Verification")
print("=" * 70)

target_files, target_duplicates = collect(ROOT / target)

print(f"\nTarget: {target}")
print(f"Images: {len(target_files)}")
print(f"Duplicate keys: {len(target_duplicates)}")

all_keys = {}

for condition in conditions:

    print("\n" + "-" * 70)
    print(f"Checking: {condition} -> {target}")
    print("-" * 70)

    input_files, input_duplicates = collect(ROOT / condition)

    missing_targets = sorted(set(input_files) - set(target_files))
    missing_inputs = sorted(set(target_files) - set(input_files))

    matched = sorted(set(input_files) & set(target_files))

    print(f"Input images:      {len(input_files)}")
    print(f"Matched pairs:     {len(matched)}")
    print(f"Missing targets:   {len(missing_targets)}")
    print(f"Missing inputs:    {len(missing_inputs)}")
    print(f"Duplicate inputs:  {len(input_duplicates)}")

    # Check dimensions of matched pairs
    dimension_errors = []

    for key in matched:
        inp = input_files[key]
        tgt = target_files[key]

        try:
            with Image.open(inp) as a:
                input_size = a.size

            with Image.open(tgt) as b:
                target_size = b.size

            if input_size != target_size:
                dimension_errors.append(
                    (key, input_size, target_size)
                )

        except Exception as e:
            dimension_errors.append((key, "READ_ERROR", str(e)))

    print(f"Dimension/read errors: {len(dimension_errors)}")

    all_keys[condition] = set(matched)


print("\n" + "=" * 70)
print("CROSS-CONDITION OVERLAP")
print("=" * 70)

common = (
    all_keys["low_blur"]
    & all_keys["low_blur_noise"]
    & all_keys["low_sharp"]
)

print(f"\nKeys common to ALL THREE classes: {len(common)}")

for condition in conditions:
    print(
        f"{condition}: {len(all_keys[condition])} valid matched keys"
    )

print("\nPairwise overlaps:")

for a, b in [
    ("low_blur", "low_blur_noise"),
    ("low_blur", "low_sharp"),
    ("low_blur_noise", "low_sharp"),
]:
    overlap = all_keys[a] & all_keys[b]
    print(f"{a} <-> {b}: {len(overlap)}")

print("\nVerification complete.")
