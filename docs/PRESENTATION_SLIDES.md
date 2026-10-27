# 📊 PlantCare AI: Presentation Slides Deck & Speaker Notes

This outline is structured for a **10-slide academic project presentation** (approx. 8–10 minutes) with accompanying speaker notes for each slide.

---

### Slide 1: Title & Introduction
- **Slide Title**: PlantCare AI: Plant Disease Detection Using Convolutional Neural Networks
- **Subtitle**: Automated Foliar Pathology Identification Powered by MobileNetV2 Transfer Learning
- **Presenter**: [Your Name]
- **Department**: [Department of Computer Science & Engineering]
- **Date**: October 2026
- **Speaker Notes**:
  > *"Good morning respected evaluators and professors. Today, I am excited to present 'PlantCare AI'—an intelligent deep learning diagnostic tool developed to identify tomato leaf diseases from photographs using MobileNetV2 Transfer Learning and Streamlit."*

---

### Slide 2: Problem Statement & Motivation
- **Key Points**:
  - Tomatoes are a vital global crop highly vulnerable to fungal, bacterial, and oomycete pathogens.
  - Delayed identification leads to severe harvest loss and excessive chemical pesticide usage.
  - Manual visual inspection by agronomists is labor-intensive and geographically constrained.
  - Early-stage leaf lesions share overlapping visual symptoms that confuse human observers.
- **Speaker Notes**:
  > *"Foliar diseases such as early blight, late blight, and bacterial spot cost the agricultural economy billions of dollars annually. Smallholder farmers often lack access to timely expert diagnosis, leading to reactive blanket chemical spraying. Our goal was to build a reproducible, lightweight computer vision system to automate this inspection."*

---

### Slide 3: Project Objectives
- **Key Points**:
  1. Build a verified 5-class tomato leaf dataset from the PlantVillage repository.
  2. Implement an end-to-end data augmentation and preprocessing pipeline.
  3. Train a lightweight MobileNetV2 Transfer Learning architecture with regularization.
  4. Perform rigorous evaluation on an untouched 15% held-out test dataset.
  5. Deploy a real-time, interactive Streamlit web application with confidence transparency.
- **Speaker Notes**:
  > *"Rather than relying on black-box predictions, we structured the project around rigorous engineering: deterministic dataset splitting, zero data leakage, empirical evaluation on held-out test data, and transparent probability breakdowns."*

---

### Slide 4: Dataset Overview & Verification
- **Key Points**:
  - Source: Peer-reviewed PlantVillage benchmark dataset.
  - Total Sample Count: **7,579 verified RGB color images**.
  - Classes:
    - Tomato Healthy (1,591 images)
    - Bacterial Spot (2,127 images)
    - Early Blight (1,000 images)
    - Late Blight (1,909 images)
    - Leaf Mold (952 images)
  - Split: Deterministic 70% Train (5,303), 15% Validation (1,135), 15% Held-Out Test (1,141).
- **Speaker Notes**:
  > *"We verified all 7,579 images for byte-level integrity using an automated verification script before training. We partitioned the dataset into 70% train, 15% validation, and 15% test. Crucially, the 1,141 test images remained completely untouched until final evaluation."*

---

### Slide 5: Methodology & Transfer Learning Architecture
- **Key Points**:
  - **Feature Extractor**: MobileNetV2 pre-trained on ImageNet (154 frozen layers).
  - **Depthwise Separable Convolutions**: Reduces computational cost ($FLOPs$) by $\approx 85\%$ compared to standard convolutions.
  - **Regularization**:
    - Embedded dynamic data augmentation (flip, rotation, zoom, contrast).
    - Global Average Pooling 2D + Batch Normalization.
    - Dropout (0.30 and 0.20) to eliminate neuron co-adaptation.
  - **Classification Head**: Dense(128, ReLU) $\rightarrow$ Dense(5, Softmax).
- **Speaker Notes**:
  > *"Instead of training a CNN from scratch—which requires massive GPU clusters and risks overfitting—we used MobileNetV2 Transfer Learning. MobileNetV2 uses depthwise separable convolutions, keeping our final model size down to just 11.6 MB while leveraging visual features learned from 1.4 million ImageNet images."*

---

### Slide 6: Model Training & Optimization
- **Key Points**:
  - Optimizer: Adam ($\alpha = 0.001$, $\beta_1=0.9, \beta_2=0.999$).
  - Loss: Sparse Categorical Cross-Entropy.
  - Callbacks: ModelCheckpoint (best validation accuracy), EarlyStopping, ReduceLROnPlateau.
  - Training Accuracy at Epoch 8: **92.16%**
  - Validation Accuracy at Best Epoch (Epoch 7): **91.98%**
  - No Overfitting: Train and Validation losses converged smoothly.
- **Speaker Notes**:
  > *"Training ran across 8 epochs with a batch size of 32. Our callbacks saved the optimal model weights at Epoch 7 where validation accuracy reached 91.98%. The close alignment between training and validation metrics proves that data augmentation and dropout successfully controlled overfitting."*

---

### Slide 7: Empirical Results on Held-Out Test Data
- **Key Points**:
  - Evaluated on **1,141 unseen test images**:
    - **Overall Test Accuracy**: **93.69%**
    - **Weighted Precision**: **93.92%**
    - **Weighted Recall**: **93.69%**
    - **Weighted F1-Score**: **93.32%**
  - Per-Class F1-Scores:
    - Healthy: **97.54%**
    - Bacterial Spot: **97.37%**
    - Leaf Mold: **94.31%**
    - Late Blight: **92.98%**
    - Early Blight: **77.60%**
- **Speaker Notes**:
  > *"On our strictly held-out test split, the model achieved 93.69% accuracy. Healthy leaves, Bacterial Spot, and Leaf Mold all surpassed 94% F1-score. Early Blight achieved high precision (97%) with lower recall (65%), as initial microscopic lesions resemble early fungal sporulation."*

---

### Slide 8: Streamlit Application Walkthrough
- **Key Points**:
  - Interactive web interface written in Python with Streamlit.
  - In-memory model caching with `@st.cache_resource` for instant inference (<40ms).
  - Robust error handling: rejects corrupt or non-image files with clear alerts.
  - Multi-class confidence bars: displays probabilities across all 5 categories.
  - Integrated Model Transparency Tab displaying actual confusion matrix and metrics.
- **Speaker Notes**:
  > *"We packaged the trained model into a Streamlit application. Users can upload any JPG or PNG leaf photo, preview it, and receive an instant diagnosis with confidence bars for all 5 classes. The app also features an evaluation tab displaying the real confusion matrix and metrics."*

---

### Slide 9: Limitations & Ethical Considerations
- **Key Points**:
  - **Controlled Environment Bias**: PlantVillage images were photographed against plain laboratory backgrounds; field illumination and soil clutter can reduce confidence.
  - **Single-Leaf Assumption**: Designed for single leaf closeups, not whole crop canopies.
  - **Responsible AI Disclaimer**: This is an educational prototype and must not replace certified agricultural extension guidance.
- **Speaker Notes**:
  > *"We believe responsible AI requires addressing limitations honestly. Because PlantVillage photos were captured in controlled lab lighting, outdoor sunlight and shadows represent a domain shift. Therefore, our application features clear educational disclaimers."*

---

### Slide 10: Conclusion & Future Scope
- **Key Points**:
  - **Summary**: Delivered a verified, reproducible 93.69% accurate leaf disease classification pipeline with an 11.6 MB footprint.
  - **Future Roadmap**:
    - Multi-leaf object detection with YOLOv8.
    - Mobile offline deployment using TensorFlow Lite (TFLite).
    - Integration of agronomic treatment guides and IPM advice.
  - **Thank you! Questions & Answers.**
- **Speaker Notes**:
  > *"In conclusion, PlantCare AI demonstrates how transfer learning can create fast, lightweight agricultural computer vision tools. In the future, we plan to convert the model to TensorFlow Lite for offline mobile use on farms. Thank you, and I welcome any questions."*
