import random
import shutil
from pathlib import Path

# ==========================================================
# Configuration
# ==========================================================

random.seed(42)

DATASET_ROOT = Path("dataset/NordTank586x371")

IMAGES_DIR = DATASET_ROOT / "images"
LABELS_DIR = DATASET_ROOT / "labels"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# ==========================================================
# Remove old split folders (if they exist)
# ==========================================================

for folder in ["train", "val", "test"]:
    split_path = DATASET_ROOT / folder
    if split_path.exists():
        shutil.rmtree(split_path)

# ==========================================================
# Collect only labelled images
# ==========================================================

images = []

for image_path in IMAGES_DIR.iterdir():

    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        continue

    label_path = LABELS_DIR / (image_path.stem + ".txt")

    if label_path.exists():
        images.append(image_path)

images.sort()
random.shuffle(images)

# ==========================================================
# Split dataset
# ==========================================================

total = len(images)

train_end = int(total * TRAIN_RATIO)
val_end = train_end + int(total * VAL_RATIO)

train_images = images[:train_end]
val_images = images[train_end:val_end]
test_images = images[val_end:]

splits = {
    "train": train_images,
    "val": val_images,
    "test": test_images,
}

# ==========================================================
# Create folders
# ==========================================================

for split_name in splits:

    (DATASET_ROOT / split_name / "images").mkdir(
        parents=True,
        exist_ok=True,
    )

    (DATASET_ROOT / split_name / "labels").mkdir(
        parents=True,
        exist_ok=True,
    )

# ==========================================================
# Copy files
# ==========================================================

for split_name, image_list in splits.items():

    for image_path in image_list:

        label_path = LABELS_DIR / f"{image_path.stem}.txt"

        shutil.copy2(
            image_path,
            DATASET_ROOT / split_name / "images" / image_path.name,
        )

        shutil.copy2(
            label_path,
            DATASET_ROOT / split_name / "labels" / label_path.name,
        )

# ==========================================================
# Summary
# ==========================================================

print("=" * 50)
print("Dataset Split Complete")
print("=" * 50)

print(f"Total Labelled Images : {total}")
print(f"Training Images       : {len(train_images)}")
print(f"Validation Images     : {len(val_images)}")
print(f"Test Images           : {len(test_images)}")

print()
print("Dataset successfully prepared for YOLO11!")