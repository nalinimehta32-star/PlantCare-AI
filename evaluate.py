"""
PlantCare AI - Model Evaluation & Metrics Pipeline
==================================================
Evaluates the trained model on the held-out test split (1,141 images).
Computes:
- Overall Accuracy
- Precision, Recall, and F1-score (macro and per-class)
- Full Classification Report
- Confusion Matrix Plot -> `outputs/confusion_matrix.png`
- Accuracy and Loss Curves -> `outputs/accuracy_loss.png`
- Summary Metrics -> `outputs/evaluation_results.json`
"""

import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")

MODEL_PATH = MODELS_DIR / "plant_disease_model.keras"
CLASS_MAPPING_PATH = MODELS_DIR / "class_indices.json"
HISTORY_PATH = OUTPUTS_DIR / "training_history.json"
TEST_MANIFEST_PATH = OUTPUTS_DIR / "test_set_manifest.json"
CONFUSION_MATRIX_PATH = OUTPUTS_DIR / "confusion_matrix.png"
ACCURACY_LOSS_PATH = OUTPUTS_DIR / "accuracy_loss.png"
RESULTS_JSON_PATH = OUTPUTS_DIR / "evaluation_results.json"


def parse_image(path_tensor, label_tensor):
    image_raw = tf.io.read_file(path_tensor)
    image = tf.io.decode_image(image_raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label_tensor


def plot_training_curves(history_data: dict, save_path: Path):
    """Plots training and validation loss and accuracy curves."""
    epochs = range(1, len(history_data["accuracy"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy subplot
    ax1.plot(epochs, [a * 100 for a in history_data["accuracy"]], "o-", color="#2980b9", label="Training Accuracy", linewidth=2)
    ax1.plot(epochs, [a * 100 for a in history_data["val_accuracy"]], "s--", color="#27ae60", label="Validation Accuracy", linewidth=2)
    ax1.set_title("Model Accuracy across Epochs", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    # Loss subplot
    ax2.plot(epochs, history_data["loss"], "o-", color="#c0392b", label="Training Loss", linewidth=2)
    ax2.plot(epochs, history_data["val_loss"], "s--", color="#e67e22", label="Validation Loss", linewidth=2)
    ax2.set_title("Model Cross-Entropy Loss across Epochs", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] Training curves saved to: {save_path.resolve()}")


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], save_path: Path):
    """Plots and saves an annotated confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Format ticks
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="PlantCare AI - Confusion Matrix (Test Split)",
        ylabel="True Category",
        xlabel="Predicted Category"
    )

    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    # Annotate numbers in cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold"
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] Confusion matrix saved to: {save_path.resolve()}")


def main():
    print("=" * 65)
    print("        PLANTCARE AI - MODEL EVALUATION PIPELINE        ")
    print("=" * 65)

    # 1. Verify existence of required artifacts
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}. Please run train.py first.")
    if not TEST_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Test manifest not found: {TEST_MANIFEST_PATH}. Please run train.py first.")

    with open(TEST_MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    with open(CLASS_MAPPING_PATH, "r") as f:
        class_to_idx = json.load(f)

    # Sort class names in index order [0, 1, 2, 3, 4]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    display_names = [c.replace("Tomato___", "").replace("_", " ") for c in class_names]

    test_paths = manifest["test_paths"]
    y_true = np.array(manifest["test_labels"])
    print(f"[INFO] Loaded {len(test_paths)} test samples across {len(class_names)} classes.")

    # 2. Build test tf.data dataset (unaugmented)
    paths_tensor = tf.constant(test_paths)
    labels_tensor = tf.constant(y_true)
    test_ds = tf.data.Dataset.from_tensor_slices((paths_tensor, labels_tensor))
    test_ds = test_ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

    # 3. Load model and predict
    print(f"[INFO] Loading model from: {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("[INFO] Running inference on held-out test dataset...")
    y_probs = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_probs, axis=1)

    # 4. Calculate metrics
    accuracy = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    clf_report_dict = classification_report(y_true, y_pred, target_names=display_names, output_dict=True, zero_division=0)
    clf_report_text = classification_report(y_true, y_pred, target_names=display_names, zero_division=0)

    print("\n" + "=" * 65)
    print("                    EVALUATION METRICS SUMMARY                    ")
    print("=" * 65)
    print(f"Test Accuracy:           {accuracy * 100:.2f}%")
    print(f"Weighted Precision:      {precision * 100:.2f}%")
    print(f"Weighted Recall:         {recall * 100:.2f}%")
    print(f"Weighted F1-Score:       {f1 * 100:.2f}%")
    print(f"Macro F1-Score:          {macro_f1 * 100:.2f}%")
    print("-" * 65)
    print("Detailed Classification Report:\n")
    print(clf_report_text)
    print("=" * 65)

    # 5. Generate Visualizations
    plot_confusion_matrix(cm, display_names, CONFUSION_MATRIX_PATH)

    if HISTORY_PATH.exists():
        with open(HISTORY_PATH, "r") as f:
            history_data = json.load(f)
        plot_training_curves(history_data, ACCURACY_LOSS_PATH)

    # 6. Save JSON results for Streamlit app
    evaluation_summary = {
        "accuracy": accuracy,
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1": float(f1),
        "macro_f1": float(macro_f1),
        "total_test_samples": len(y_true),
        "per_class": {
            name: {
                "precision": float(clf_report_dict[name]["precision"]),
                "recall": float(clf_report_dict[name]["recall"]),
                "f1_score": float(clf_report_dict[name]["f1-score"]),
                "support": int(clf_report_dict[name]["support"])
            }
            for name in display_names
        }
    }

    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(evaluation_summary, f, indent=4)
    print(f"[INFO] Summary metrics saved to {RESULTS_JSON_PATH.resolve()}")
    print("[SUCCESS] Model evaluation complete!")


if __name__ == "__main__":
    main()
