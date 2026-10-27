"""
PlantCare AI - Streamlit Web Application
========================================
A beginner-friendly, interactive web application for detecting tomato leaf diseases
using Deep Learning (MobileNetV2 Transfer Learning).
"""

import json
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError
import streamlit as st
import tensorflow as tf

# Page configuration
st.set_page_config(
    page_title="PlantCare AI - Tomato Leaf Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Light Theme CSS Styling
st.markdown(
    """
    <style>
        /* Main background and clean typography */
        .main {
            background-color: #FFFFFF;
            color: #1F2937;
        }
        
        /* Metric Cards Styling */
        div[data-testid="stMetricValue"] {
            font-size: 1.75rem;
            font-weight: 700;
            color: #1E3A8A;
        }
        div[data-testid="stMetricLabel"] {
            color: #4B5563;
            font-weight: 500;
        }
        
        /* Clean File Uploader Area */
        div[data-testid="stFileUploader"] {
            background-color: #F8FAFC;
            border: 1px dashed #CBD5E1;
            border-radius: 8px;
            padding: 10px;
        }

        /* Tab Navigation Styling */
        button[data-baseweb="tab"] {
            font-weight: 600;
            color: #4B5563;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #2563EB !important;
            border-bottom-color: #2563EB !important;
        }
        
        /* Data Tables */
        .stTable {
            background-color: #FFFFFF;
            border-radius: 8px;
            border: 1px solid #E5E7EB;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Paths
MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")
MODEL_PATH = MODELS_DIR / "plant_disease_model.keras"
CLASS_MAPPING_PATH = MODELS_DIR / "class_indices.json"
RESULTS_PATH = OUTPUTS_DIR / "evaluation_results.json"
CM_PATH = OUTPUTS_DIR / "confusion_matrix.png"
CURVES_PATH = OUTPUTS_DIR / "accuracy_loss.png"
DISTRIBUTION_PATH = OUTPUTS_DIR / "class_distribution.png"

# Supported classes friendly display
FRIENDLY_NAMES = {
    "Tomato___Bacterial_spot": "Tomato Bacterial Spot",
    "Tomato___Early_blight": "Tomato Early Blight",
    "Tomato___healthy": "Tomato Healthy",
    "Tomato___Late_blight": "Tomato Late Blight",
    "Tomato___Leaf_Mold": "Tomato Leaf Mold"
}

IMG_SIZE = (224, 224)


@st.cache_resource(show_spinner="Loading trained Deep Learning model...")
def load_trained_model():
    """Loads the serialized Keras model once and caches it in memory."""
    if not MODEL_PATH.exists():
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


@st.cache_data
def load_class_mapping():
    """Loads class-to-index mapping from disk."""
    if not CLASS_MAPPING_PATH.exists():
        return None
    with open(CLASS_MAPPING_PATH, "r") as f:
        class_to_idx = json.load(f)
    # Reverse mapping: index -> class name
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return idx_to_class


@st.cache_data
def load_evaluation_metrics():
    """Loads actual test set evaluation results."""
    if not RESULTS_PATH.exists():
        return None
    with open(RESULTS_PATH, "r") as f:
        return json.load(f)


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    Preprocesses an uploaded PIL Image consistently with training pipeline:
    - Converts to RGB (3 channels)
    - Resizes to (224, 224)
    - Converts to float32 numpy array
    - Adds batch dimension: (1, 224, 224, 3)
    """
    rgb_img = pil_image.convert("RGB")
    resized_img = rgb_img.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    img_array = np.array(resized_img, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch


# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/plant-under-sun.png", width=70)
    st.title("PlantCare AI")
    st.markdown("**Plant Disease Detection Using CNN**")
    st.markdown("---")

    st.subheader("📋 Supported Tomato Classes")
    st.markdown(
        """
        - 🟢 **Tomato Healthy**
        - 🔴 **Tomato Bacterial Spot**
        - 🟠 **Tomato Early Blight**
        - 🟤 **Tomato Late Blight**
        - 🟡 **Tomato Leaf Mold**
        """
    )

    st.markdown("---")
    st.info(
        "💡 **Tip**: For best diagnostic accuracy, use clear, well-lit photos "
        "of a single leaf against a neutral or natural background."
    )

    st.markdown("---")
    st.caption("Developed with TensorFlow, Keras & Streamlit.")


# Main UI Header
st.title("🌿 PlantCare AI: Plant Disease Detection")
st.markdown(
    "An educational deep learning web application powered by **MobileNetV2 Transfer Learning** "
    "to identify common diseases affecting tomato foliage."
)

# Tab navigation
tab_diag, tab_metrics, tab_about = st.tabs([
    "🔬 Leaf Diagnosis",
    "📊 Model Performance & Metrics",
    "ℹ️ About & How It Works"
])

# -------------------------------------------------------------
# TAB 1: Leaf Diagnosis
# -------------------------------------------------------------
with tab_diag:
    st.subheader("Upload a Tomato Leaf Photo")

    model = load_trained_model()
    idx_to_class = load_class_mapping()

    if model is None or idx_to_class is None:
        st.warning(
            "⚠️ **Trained model not found!**\n\n"
            "Please ensure `models/plant_disease_model.keras` exists. "
            "Run `python train.py` in the terminal to train and save the model."
        )
    else:
        uploaded_file = st.file_uploader(
            "Choose a leaf image file (JPG, JPEG, or PNG)",
            type=["jpg", "jpeg", "png"],
            help="Select a clear photograph of a tomato leaf."
        )

        col_img, col_pred = st.columns([1, 1], gap="large")

        if uploaded_file is not None:
            try:
                pil_image = Image.open(uploaded_file)
                # Verify image can be read
                pil_image.verify()
                # Reopen after verify
                uploaded_file.seek(0)
                pil_image = Image.open(uploaded_file)

                with col_img:
                    st.image(
                        pil_image,
                        caption=f"Uploaded: {uploaded_file.name}",
                        use_container_width=True
                    )
                    st.caption(f"Image Resolution: {pil_image.size[0]} x {pil_image.size[1]} pixels | Format: {pil_image.format}")

                with col_pred:
                    st.markdown("### Disease Analysis")
                    analyze_btn = st.button("🔍 Analyze Leaf Image", type="primary", use_container_width=True)

                    if analyze_btn:
                        with st.spinner("Analyzing leaf patterns using MobileNetV2..."):
                            # Preprocess
                            processed_input = preprocess_image(pil_image)

                            # Predict
                            predictions = model.predict(processed_input, verbose=0)[0]
                            top_idx = int(np.argmax(predictions))
                            top_prob = float(predictions[top_idx])
                            top_raw_class = idx_to_class[top_idx]
                            top_friendly_name = FRIENDLY_NAMES.get(top_raw_class, top_raw_class)

                            is_healthy = "healthy" in top_raw_class.lower()

                        # Display Top Prediction Badge
                        if is_healthy:
                            st.success(f"### Result: {top_friendly_name}")
                            st.markdown(f"**Confidence Score:** `{top_prob * 100:.2f}%`")
                            st.markdown("✅ **Status: Healthy Plant Leaf**")
                            st.write("No major fungal or bacterial pathogen signatures were detected.")
                        else:
                            st.error(f"### Result: {top_friendly_name}")
                            st.markdown(f"**Confidence Score:** `{top_prob * 100:.2f}%`")
                            st.markdown("⚠️ **Status: Disease Detected**")
                            st.write(f"The model detected visual symptoms characteristic of **{top_friendly_name}**.")

                        st.markdown("---")
                        st.markdown("#### Confidence Breakdown Across All Categories:")

                        # Sorted classes by probability
                        sorted_indices = np.argsort(predictions)[::-1]
                        for i in sorted_indices:
                            raw_name = idx_to_class[i]
                            friendly = FRIENDLY_NAMES.get(raw_name, raw_name)
                            score = float(predictions[i])
                            st.write(f"**{friendly}**: `{score * 100:.2f}%`")
                            st.progress(min(score, 1.0))

            except (UnidentifiedImageError, Exception) as e:
                st.error(f"❌ **Invalid Image File**: The uploaded file could not be decoded as an image. Details: {e}")
        else:
            with col_img:
                st.info("👆 Please upload a tomato leaf image above to begin.")

    # Agricultural disclaimer callout
    st.markdown("---")
    st.warning(
        "⚠️ **Educational Disclaimer**: PlantCare AI is an experimental image classification prototype "
        "trained on the PlantVillage benchmark dataset. Predictions depend heavily on image quality, illumination, "
        "and camera angles. This application does not constitute a certified agricultural diagnostic service. "
        "Never make critical farming or chemical management decisions based solely on automated predictions."
    )

# -------------------------------------------------------------
# TAB 2: Model Performance & Metrics
# -------------------------------------------------------------
with tab_metrics:
    st.subheader("Empirical Model Performance on Held-Out Test Set")
    st.markdown(
        "These metrics represent the model's actual performance evaluated on **1,141 untouched test images** "
        "(15% held-out test split) that were never seen during training or validation."
    )

    metrics_data = load_evaluation_metrics()

    if metrics_data:
        # High-level metric cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overall Test Accuracy", f"{metrics_data['accuracy'] * 100:.2f}%")
        m2.metric("Weighted Precision", f"{metrics_data['weighted_precision'] * 100:.2f}%")
        m3.metric("Weighted Recall", f"{metrics_data['weighted_recall'] * 100:.2f}%")
        m4.metric("Weighted F1-Score", f"{metrics_data['weighted_f1'] * 100:.2f}%")

        st.markdown("---")
        st.subheader("Per-Class Classification Report")

        # Per-class table
        table_rows = []
        for cname, stats in metrics_data.get("per_class", {}).items():
            table_rows.append({
                "Category": cname,
                "Precision (%)": f"{stats['precision'] * 100:.2f}%",
                "Recall (%)": f"{stats['recall'] * 100:.2f}%",
                "F1-Score (%)": f"{stats['f1_score'] * 100:.2f}%",
                "Test Support (Samples)": stats["support"]
            })
        st.table(table_rows)

        st.markdown("---")
        st.subheader("Evaluation Visualizations")
        col_cm, col_curves = st.columns(2)

        with col_cm:
            st.markdown("#### Confusion Matrix")
            if CM_PATH.exists():
                st.image(str(CM_PATH), caption="Confusion Matrix on Held-Out Test Data", use_container_width=True)
            else:
                st.info("Confusion matrix plot not found. Run `python evaluate.py` to generate it.")

        with col_curves:
            st.markdown("#### Training & Validation History")
            if CURVES_PATH.exists():
                st.image(str(CURVES_PATH), caption="Accuracy & Cross-Entropy Loss Across 8 Epochs", use_container_width=True)
            else:
                st.info("Training curve plot not found. Run `python evaluate.py` to generate it.")

        if DISTRIBUTION_PATH.exists():
            st.markdown("---")
            st.markdown("#### Dataset Class Distribution")
            st.image(str(DISTRIBUTION_PATH), caption="PlantVillage Tomato Class Distribution", use_container_width=True)

    else:
        st.info(
            "Evaluation metrics have not been generated yet. "
            "Please run `python evaluate.py` to evaluate the saved model."
        )

# -------------------------------------------------------------
# TAB 3: About & How It Works
# -------------------------------------------------------------
with tab_about:
    st.subheader("About PlantCare AI")
    st.markdown(
        """
        **PlantCare AI** is designed to demonstrate how modern Convolutional Neural Networks (CNNs)
        and Transfer Learning can assist in identifying plant leaf pathologies.

        ### 🏗️ How Transfer Learning Works:
        1. **Feature Extraction**: We utilize **MobileNetV2**, an architecture pre-trained on ImageNet (over 1.4 million real-world photographs).
           The network's earlier convolutional filters are adept at detecting edges, corners, circular contours, textures, and color variations.
        2. **Fine-Tuning Head**: We frozen the pretrained convolutional base and attached custom dense layers:
           - Global Average Pooling (aggregates 7x7 spatial maps into a compact 1,280-dimension vector)
           - Batch Normalization & Dropout (30% drop rate to curb overfitting)
           - Dense Layer (128 units with ReLU activation)
           - Output Layer (5 units with Softmax activation for categorical probability distribution)
        3. **Data Augmentation**: During training, random horizontal flips, rotations (±10%), zooms (±10%), and contrast shifts were applied
           dynamically to teach the model invariance against camera angles and varying lighting.

        ### 📚 Dataset Attribution
        - **Source**: PlantVillage Dataset (Hughes et al., 2015).
        - **Citation**: *Hughes, D., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. arXiv:1511.08060.*
        - **License**: Public Domain / Creative Commons Attribution (CC-BY).

        ### ⚠️ System Limitations:
        - **Lab Background Bias**: The benchmark PlantVillage dataset was photographed under controlled laboratory lighting with uniform backgrounds.
          Photos taken in direct outdoor sunlight or with cluttered soil backgrounds may exhibit reduced confidence.
        - **Symptom Overlap**: Early blight in its initial lesion phase can occasionally resemble early bacterial spots.
        - **Single-Leaf Assumption**: The model is optimized for single-leaf closeups, not whole plant canopies.
        """
    )
