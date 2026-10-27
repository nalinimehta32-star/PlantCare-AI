"""
PlantCare AI - Dataset Inspection & Verification Script
======================================================
This script scans the `data/` directory for tomato leaf disease images,
validates that all files are uncorrupted images, counts images per category,
and generates a class-distribution bar plot saved in `outputs/`.
"""

import os
import sys
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Supported target categories
TARGET_CLASSES = [
    "Tomato healthy",
    "Tomato bacterial spot",
    "Tomato early blight",
    "Tomato late blight",
    "Tomato leaf mold"
]

# Common aliases/naming variations found in PlantVillage dataset archives
CLASS_ALIASES = {
    "Tomato healthy": ["tomato___healthy", "tomato_healthy", "healthy"],
    "Tomato bacterial spot": ["tomato___bacterial_spot", "tomato_bacterial_spot", "bacterial_spot"],
    "Tomato early blight": ["tomato___early_blight", "tomato_early_blight", "early_blight"],
    "Tomato late blight": ["tomato___late_blight", "tomato_late_blight", "late_blight"],
    "Tomato leaf mold": ["tomato___leaf_mold", "tomato_leaf_mold", "leaf_mold"]
}

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def match_folder_to_class(folder_name: str) -> str | None:
    """Matches a directory name against the 5 target tomato disease classes."""
    cleaned = folder_name.strip().lower().replace("-", "_").replace(" ", "_")
    for canonical_name, aliases in CLASS_ALIASES.items():
        for alias in aliases:
            if alias in cleaned:
                return canonical_name
    return None


def inspect_dataset(data_dir: Path):
    print("=" * 65)
    print("      PLANTCARE AI - DATASET VERIFICATION & INSPECTION      ")
    print("=" * 65)
    print(f"Inspecting directory: {data_dir.resolve()}\n")

    if not data_dir.exists():
        print(f"[ERROR] Directory does not exist: {data_dir}")
        return False

    # Find all subdirectories that might represent classes
    subdirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]

    # If the user extracted a nested folder (e.g. data/PlantVillage/...)
    if len(subdirs) == 1 and match_folder_to_class(subdirs[0].name) is None:
        nested_dir = subdirs[0]
        nested_subdirs = [d for d in nested_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        if nested_subdirs:
            print(f"[NOTE] Found nested folder '{nested_dir.name}', scanning inside it...")
            subdirs = nested_subdirs

    if not subdirs:
        print("[WARNING] No class subdirectories found in the data/ folder.")
        print("\nPlease structure your dataset under `data/` as follows:")
        for cls in TARGET_CLASSES:
            print(f"  data/{cls.replace(' ', '_')}/")
        print("\nOr keep the original PlantVillage names (e.g., data/Tomato___healthy/).")
        return False

    print(f"Discovered {len(subdirs)} subdirectories:")
    for s in subdirs:
        matched = match_folder_to_class(s.name)
        status = f"-> Matched to '{matched}'" if matched else "(Not one of the 5 target classes, skipped)"
        print(f"  - {s.name:<35} {status}")

    print("\n" + "-" * 65)
    print("Verifying image integrity...")
    print("-" * 65)

    stats = {cls: {"valid": 0, "corrupt": 0, "unsupported": 0} for cls in TARGET_CLASSES}
    matched_folders = {}

    for s in subdirs:
        matched = match_folder_to_class(s.name)
        if not matched:
            continue
        matched_folders[matched] = s

        for file_path in s.iterdir():
            if file_path.is_file():
                if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    stats[matched]["unsupported"] += 1
                    continue

                # Verify image can be opened and is not corrupt
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                    stats[matched]["valid"] += 1
                except Exception as e:
                    stats[matched]["corrupt"] += 1
                    print(f"[CORRUPT IMAGE] {file_path.name}: {e}")

    # Summary report
    print("\n" + "=" * 65)
    print(f"{'Category':<28} | {'Valid':<8} | {'Corrupt':<8} | {'Unsupported':<12}")
    print("-" * 65)

    all_present = True
    total_valid = 0

    for cls in TARGET_CLASSES:
        v = stats[cls]["valid"]
        c = stats[cls]["corrupt"]
        u = stats[cls]["unsupported"]
        total_valid += v
        print(f"{cls:<28} | {v:<8} | {c:<8} | {u:<12}")
        if v == 0:
            all_present = False

    print("-" * 65)
    print(f"Total valid images ready for training: {total_valid}")
    print("=" * 65)

    # Generate visualization if data is present
    if total_valid > 0:
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(exist_ok=True)
        chart_path = outputs_dir / "class_distribution.png"

        categories = [cls.replace("Tomato ", "") for cls in TARGET_CLASSES]
        counts = [stats[cls]["valid"] for cls in TARGET_CLASSES]
        colors = ["#2ecc71" if "healthy" in c.lower() else "#e74c3c" for c in categories]

        plt.figure(figsize=(9, 5))
        bars = plt.bar(categories, counts, color=colors, edgecolor="black", alpha=0.85)
        plt.title("PlantCare AI - Class Distribution (Tomato Categories)", fontsize=14, fontweight="bold")
        plt.xlabel("Category", fontsize=11)
        plt.ylabel("Number of Images", fontsize=11)
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            plt.annotate(f"{height}",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha="center", va="bottom", fontsize=10, fontweight="bold")

        plt.tight_layout()
        plt.savefig(chart_path, dpi=300)
        plt.close()
        print(f"\n[SAVED] Class distribution chart saved to: {chart_path.resolve()}")

    if not all_present:
        missing = [cls for cls in TARGET_CLASSES if stats[cls]["valid"] == 0]
        print(f"\n[ACTION REQUIRED] The following categories have 0 images:")
        for m in missing:
            print(f"  - {m}")
        return False

    print("\n[SUCCESS] All 5 tomato categories are present and verified!")
    return True


if __name__ == "__main__":
    data_path = Path("data")
    success = inspect_dataset(data_path)
    sys.exit(0 if success else 1)
