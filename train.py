"""
PlantCare AI - Model Training Pipeline
======================================
Architecture: MobileNetV2 Transfer Learning
- Pretrained weights: ImageNet
- Input shape: (224, 224, 3)
- Target: 5 Tomato Leaf Disease Classes
- Splits: 70% Train, 15% Validation, 15% Held-out Test
- Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
"""

import os
import json
import random
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Hyperparameters & Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 8
LEARNING_RATE = 1e-3

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")

MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

MODEL_SAVE_PATH = MODELS_DIR / "plant_disease_model.keras"
CLASS_MAPPING_PATH = MODELS_DIR / "class_indices.json"
HISTORY_SAVE_PATH = OUTPUTS_DIR / "training_history.json"
TEST_MANIFEST_PATH = OUTPUTS_DIR / "test_set_manifest.json"


def scan_and_split_dataset(data_dir: Path, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Scans the dataset directory, collects all valid images per class,
    and produces a deterministic 70/15/15 split without data leakage.
    """
    print(f"[INFO] Scanning directory: {data_dir.resolve()}...")
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    
    if not class_dirs:
        raise ValueError(f"No class folders found in {data_dir}!")

    classes = [d.name for d in class_dirs]
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []

    print(f"[INFO] Discovered {len(classes)} classes:")
    for cls_name in classes:
        cls_dir = data_dir / cls_name
        valid_images = [
            f for f in cls_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
        # Deterministic shuffle per class
        valid_images.sort()
        random.Random(SEED).shuffle(valid_images)

        n = len(valid_images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        cls_train = valid_images[:n_train]
        cls_val = valid_images[n_train:n_train + n_val]
        cls_test = valid_images[n_train + n_val:]

        train_paths.extend([str(p) for p in cls_train])
        train_labels.extend([class_to_idx[cls_name]] * len(cls_train))

        val_paths.extend([str(p) for p in cls_val])
        val_labels.extend([class_to_idx[cls_name]] * len(cls_val))

        test_paths.extend([str(p) for p in cls_test])
        test_labels.extend([class_to_idx[cls_name]] * len(cls_test))

        print(f"  - {cls_name:<30}: {n} total (Train: {len(cls_train)}, Val: {len(cls_val)}, Test: {len(cls_test)})")

    # Save class mapping
    with open(CLASS_MAPPING_PATH, "w") as f:
        json.dump(class_to_idx, f, indent=4)
    print(f"[INFO] Saved class mapping to {CLASS_MAPPING_PATH}")

    # Save test manifest for evaluate.py
    with open(TEST_MANIFEST_PATH, "w") as f:
        json.dump({
            "test_paths": test_paths,
            "test_labels": test_labels,
            "classes": classes
        }, f, indent=4)
    print(f"[INFO] Saved held-out test manifest to {TEST_MANIFEST_PATH}")

    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels), classes


def parse_image(path_tensor, label_tensor):
    """Reads and decodes an image file into a tensor resized to (224, 224, 3)."""
    image_raw = tf.io.read_file(path_tensor)
    image = tf.io.decode_image(image_raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label_tensor


def create_tf_dataset(paths, labels, is_training=False):
    """Builds an optimized tf.data.Dataset pipeline with shuffling and prefetching."""
    paths_tensor = tf.constant(paths)
    labels_tensor = tf.constant(labels)

    dataset = tf.data.Dataset.from_tensor_slices((paths_tensor, labels_tensor))
    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(paths), 2000), seed=SEED)

    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def build_model(num_classes: int) -> tf.keras.Model:
    """
    Constructs the MobileNetV2 Transfer Learning architecture.
    Augmentation is embedded inside the model and executes ONLY during training.
    """
    inputs = tf.keras.Input(shape=(224, 224, 3), name="input_image")

    # Data Augmentation (applied only during model.fit via training=True)
    x = tf.keras.layers.RandomFlip("horizontal", seed=SEED)(inputs)
    x = tf.keras.layers.RandomRotation(0.10, seed=SEED)(x)
    x = tf.keras.layers.RandomZoom(0.10, seed=SEED)(x)
    x = tf.keras.layers.RandomContrast(0.10, seed=SEED)(x)

    # MobileNetV2 expected normalization [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # Pretrained MobileNetV2 base (frozen feature extractor)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.30, seed=SEED, name="dropout_1")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_128")(x)
    x = tf.keras.layers.Dropout(0.20, seed=SEED, name="dropout_2")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="PlantCare_MobileNetV2")
    return model


def main():
    print("=" * 65)
    print("        PLANTCARE AI - MODEL TRAINING PIPELINE        ")
    print("=" * 65)

    # 1. Prepare data splits
    (train_p, train_l), (val_p, val_l), (test_p, test_l), classes = scan_and_split_dataset(DATA_DIR)
    num_classes = len(classes)

    print(f"\n[INFO] Dataset Summary:")
    print(f"  - Training samples:   {len(train_p)} (70%)")
    print(f"  - Validation samples: {len(val_p)} (15%)")
    print(f"  - Test samples:       {len(test_p)} (15%)")

    # 2. Build tf.data datasets
    train_ds = create_tf_dataset(train_p, train_l, is_training=True)
    val_ds = create_tf_dataset(val_p, val_l, is_training=False)

    # 3. Build & compile model
    print("\n[INFO] Building MobileNetV2 Transfer Learning Architecture...")
    model = build_model(num_classes)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 4. Define Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_SAVE_PATH),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.2,
            patience=2,
            min_lr=1e-5,
            verbose=1
        )
    ]

    # 5. Train Model
    print("\n" + "=" * 65)
    print(f"Starting training for {INITIAL_EPOCHS} epochs with batch size {BATCH_SIZE}...")
    print("=" * 65 + "\n")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        callbacks=callbacks
    )

    # 6. Save Training History
    history_dict = {
        "accuracy": [float(v) for v in history.history.get("accuracy", [])],
        "val_accuracy": [float(v) for v in history.history.get("val_accuracy", [])],
        "loss": [float(v) for v in history.history.get("loss", [])],
        "val_loss": [float(v) for v in history.history.get("val_loss", [])],
        "lr": [float(v) for v in history.history.get("learning_rate", [])] if "learning_rate" in history.history else []
    }

    with open(HISTORY_SAVE_PATH, "w") as f:
        json.dump(history_dict, f, indent=4)
    print(f"\n[INFO] Training history saved to {HISTORY_SAVE_PATH}")
    print(f"[SUCCESS] Model training complete! Best model saved to: {MODEL_SAVE_PATH.resolve()}")


if __name__ == "__main__":
    main()
