# 📄 PlantCare AI: Comprehensive Academic Project Report

**Project Title**: PlantCare AI: Plant Disease Detection Using Convolutional Neural Networks (CNN)  
**Author**: [Student Name / Placeholder]  
**Degree / Department**: [B.Tech / B.Sc / M.Tech in Computer Science / Artificial Intelligence]  
**Academic Year**: 2026  
**Technologies**: Python 3.11, TensorFlow 2.21, Keras 3.15, MobileNetV2, Streamlit, Scikit-learn, Matplotlib  

---

## 1. Abstract

Plant diseases represent a leading cause of global agricultural yield loss, threatening food security and farmer livelihoods. Early, precise diagnosis of foliar pathologies is essential for timely, targeted intervention. This project develops **PlantCare AI**, an intelligent, accessible diagnostic system that uses deep learning and computer vision to identify diseases from photographs of tomato leaves (*Solanum lycopersicum*). Utilizing **MobileNetV2 Transfer Learning**, the model was trained and evaluated on 7,579 color images across five canonical classes from the benchmark PlantVillage dataset (Tomato Healthy, Bacterial Spot, Early Blight, Late Blight, and Leaf Mold). Through data augmentation, batch normalization, and dropout regularization, the architecture achieved an empirical **Test Accuracy of 93.69%**, a **Weighted F1-score of 93.32%**, and an inference latency under 40 milliseconds on CPU. The model was deployed inside an interactive web application built with **Streamlit**, featuring instant visual diagnosis, multi-class confidence breakdowns, and comprehensive evaluation summaries. This report details the system architecture, preprocessing pipeline, experimental methodology, evaluation results, and academic discussion.

---

## 2. Introduction & Background

Agriculture faces unprecedented challenges from climatic variability, pathogen evolution, and resource scarcity. Tomatoes are one of the world's most widely consumed and economically significant horticultural crops. However, tomato foliage is susceptible to a broad spectrum of bacterial, fungal, and oomycete infections that cause defoliation, blighting, and total harvest failure.

Traditionally, crop disease identification relies on manual visual inspection by agricultural extension officers or laboratory pathotyping (such as polymerase chain reaction or culturing). These manual methods are labor-intensive, geographically constrained, and often slow, leading farmers to either under-react or indiscriminately apply chemical fungicides and bactericides. 

Computer Vision (CV) powered by Deep Convolutional Neural Networks (CNNs) offers an automated, non-invasive alternative capable of analyzing microscopic and macroscopic leaf lesions in milliseconds. By packaging these deep neural networks into user-friendly web interfaces, even non-expert growers and students can inspect leaf health.

---

## 3. Problem Statement

Existing manual disease diagnosis in tomato farming suffers from:
1. **Scarcity of Expert Pathologists**: Smallholder farmers often lack immediate access to certified agronomists.
2. **Visual Symptom Confusion**: Early stages of early blight, late blight, and bacterial spots present subtle, overlapping necrotic chlorosis that is difficult for untrained observers to distinguish.
3. **Over-reliance on Chemical Sprays**: Delayed diagnosis leads to reactive, blanket spraying of broad-spectrum agrochemicals, damaging environmental ecosystems and increasing input costs.

**PlantCare AI** directly addresses this challenge by providing an open, reproducible, and accurate automated diagnostic tool.

---

## 4. Objectives

1. Assemble and rigorously validate a benchmark dataset of 5 distinct tomato foliar conditions from the PlantVillage repository.
2. Build an image preprocessing and dynamic data augmentation pipeline to handle varying angles, rotations, and contrast levels.
3. Train a lightweight, high-performance deep convolutional neural network via **MobileNetV2 Transfer Learning** with frozen ImageNet feature weights and a custom classification head.
4. Evaluate performance on an untouched held-out test split (15%) using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
5. Package the model into an intuitive, real-time web application using Streamlit that provides prediction confidence and transparency.

---

## 5. Literature Review & Theoretical Context

### 5.1 Evolution of Deep Learning in Agriculture
Early agricultural computer vision relied on handcrafted feature engineering, such as Gray-Level Co-occurrence Matrix (GLCM) for texture extraction and Color Histograms, combined with Support Vector Machines (SVM) or Random Forests. While effective in controlled laboratory settings, these methods broke down under varying field illumination, complex backgrounds, and occlusion.

In 2015, Hughes and Salathé released the **PlantVillage** dataset, demonstrating that deep CNNs could classify 26 diseases across 14 crop species with over 99% laboratory accuracy. Subsequent studies (Mohanty et al., 2016; Ferentinos, 2018) highlighted the power of transfer learning architectures—such as VGG, ResNet, and MobileNet—over networks trained from scratch, citing faster convergence and resilience against overfitting.

### 5.2 MobileNetV2 Architecture
MobileNetV2 (Sandler et al., 2018) was designed specifically for resource-constrained edge devices and web deployments. Its core innovation is the **Inverted Residual Block with Linear Bottlenecks**:
- **Depthwise Separable Convolutions**: Splits standard convolutions into a 3x3 depthwise convolution (filtering spatial patterns per channel) followed by a 1x1 pointwise convolution (combining channels). This reduces computational cost ($FLOPs$) by a factor of 8 to 9 compared to standard convolutions with minimal accuracy loss.
- **Linear Bottlenecks**: Non-linear activations (like ReLU) are removed in bottleneck layers to prevent information loss in low-dimensional manifolds.
- **Inverted Residuals**: Memory-efficient skip connections connect thin bottleneck layers rather than expanded representations.

---

## 6. Dataset Description

The project utilizes the peer-reviewed **PlantVillage Dataset**:

- **Total Selected Images**: 7,579 verified RGB color images.
- **Classes**:
  1. `Tomato___healthy`: 1,591 images (Negative control)
  2. `Tomato___Bacterial_spot`: 2,127 images (*Xanthomonas perforans*)
  3. `Tomato___Early_blight`: 1,000 images (*Alternaria solani*)
  4. `Tomato___Late_blight`: 1,909 images (*Phytophthora infestans*)
  5. `Tomato___Leaf_Mold`: 952 images (*Passalora fulva*)
- **Data Integrity**: Verified using `verify_dataset.py`, confirming 0 corrupted images and 0 unsupported formats.
- **Data Splitting**: Deterministic 70% Training (5,303 images), 15% Validation (1,135 images), and 15% Held-Out Testing (1,141 images).

---

## 7. System Architecture & Methodology

```text
[Raw Leaf Image]
       │
       ▼
[Image Preprocessing: Bilinear Resize to (224, 224), Float32 Normalization to [-1, 1]]
       │
       ▼
[Data Augmentation (Training Only): RandomFlip, RandomRotation(0.1), RandomZoom(0.1), RandomContrast(0.1)]
       │
       ▼
[MobileNetV2 Feature Extractor: 154 Pretrained Layers, Frozen ImageNet Weights]
       │
       ▼
[Global Average Pooling 2D: Converts 7x7x1280 feature maps into 1,280 feature vector]
       │
       ▼
[Batch Normalization & Dropout (0.30)]
       │
       ▼
[Dense Layer: 128 Units, ReLU Activation]
       │
       ▼
[Dropout (0.20)]
       │
       ▼
[Dense Output Layer: 5 Units, Softmax Activation]
       │
       ▼
[Probabilities Vector -> Top Predicted Class + Confidence Score]
```

### 7.1 Training Configuration
- **Optimizer**: Adam ($\beta_1 = 0.9, \beta_2 = 0.999$, initial learning rate $\alpha = 0.001$)
- **Loss Function**: Sparse Categorical Cross-Entropy:
  $$\mathcal{L} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$
- **Batch Size**: 32
- **Epochs**: 8 (with Early Stopping patience of 3 and Model Checkpoint saving the best validation accuracy)
- **Regularization**: Dropout (30% and 20%) to curb neuron co-adaptation; Batch Normalization to stabilize internal covariate shift.

---

## 8. Experimental Results & Evaluation

The final model weights (`models/plant_disease_model.keras`) were evaluated on the **1,141 held-out test images**.

### 8.1 Quantitative Performance Summary
| Metric | Value |
| :--- | :--- |
| **Overall Test Accuracy** | **93.69%** |
| **Weighted Precision** | **93.92%** |
| **Weighted Recall** | **93.69%** |
| **Weighted F1-Score** | **93.32%** |
| **Macro F1-Score** | **91.96%** |

### 8.2 Per-Class Classification Report
| Class | Precision (%) | Recall (%) | F1-Score (%) | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Tomato Bacterial Spot** | 96.33% | 98.44% | **97.37%** | 320 |
| **Tomato Early Blight** | 97.00% | 64.67% | **77.60%** | 150 |
| **Tomato Healthy** | 95.97% | 99.17% | **97.54%** | 240 |
| **Tomato Late Blight** | 89.39% | 96.86% | **92.98%** | 287 |
| **Tomato Leaf Mold** | 90.97% | 97.92% | **94.31%** | 144 |

### 8.3 Analysis of Findings
1. **High Healthy Detection (97.54% F1)**: The network reliably distinguishes healthy leaves from diseased specimens, critical for avoiding false alarms.
2. **Early Blight Recall Discrepancy**: While precision for Early Blight was high (97%), recall was 64.67%. Inspection of false negatives reveals that in early lesion stages, small circular spots closely mimic early Late Blight or Leaf Mold velvet patches. As lesions mature and develop characteristic concentric rings, classification confidence rises dramatically.
3. **Generalization Stability**: Training accuracy reached 92.16% at epoch 8, while validation accuracy reached 91.98% and test accuracy scored 93.69%. The close alignment between train, val, and test metrics demonstrates that data augmentation and dropout successfully prevented overfitting.

---

## 9. Streamlit Web Application

The front-end was developed using **Streamlit** to provide a production-ready interface:
- **Model Caching**: Leverages `@st.cache_resource` to keep the 11.6 MB model in memory, avoiding redundant disk I/O on user interactions.
- **Input Validation**: Verifies uploaded image streams using Pillow to reject corrupted files or non-image binaries before passing tensors to TensorFlow.
- **Dual Visuals**: Displays the uploaded photograph alongside prediction probability progress bars across all five categories.
- **Transparency Tab**: Integrates saved evaluation charts (`confusion_matrix.png`, `accuracy_loss.png`) directly into the interface so users can inspect real test metrics.
- **Prominent Disclaimer**: Warns users against treating the educational prototype as certified agricultural advice.

---

## 10. Limitations

1. **Controlled Lighting Bias**: PlantVillage images were captured against uniform gray/black paper in laboratory lighting. Real-world farm images with complex soil backgrounds or varying solar glare may exhibit lower confidence.
2. **Single-Leaf Requirement**: The model expects a single, centered leaf; it cannot currently segment multiple leaves on a tree canopy.
3. **Symptom Staging**: Early-stage foliar infections share visual morphology across pathogens before distinct concentric rings or mold mats form.

---

## 11. Future Scope

1. **Field Adaptation via GANs/Diffusion**: Augment training data with synthetic field backgrounds (soil, weeds, shadows) using generative models.
2. **Object Detection (YOLOv8 / Faster R-CNN)**: Transition from whole-image classification to bounding-box detection to locate and count multiple disease lesions across entire plants.
3. **Edge Deployment**: Convert the Keras model to TensorFlow Lite (TFLite) or ONNX format for offline mobile deployment on Android / iOS devices used by field agronomists.
4. **Treatment Recommendation Integration**: Incorporate an agronomic knowledge base to suggest organic remedies and IPM (Integrated Pest Management) protocols upon diagnosis.

---

## 12. Conclusion

PlantCare AI successfully demonstrates the end-to-end implementation of an educational deep learning computer vision system. By leveraging MobileNetV2 transfer learning, the system achieves **93.69% test accuracy** across 1,141 test images while maintaining a lightweight footprint of only 11.6 MB. Coupled with a responsive Streamlit interface and comprehensive evaluation metrics, the project provides an accessible, robust blueprint for agricultural AI applications.

---

## 13. References

1. Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. *arXiv preprint arXiv:1511.08060*.
2. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR)*, 4510-4520.
3. Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. *Frontiers in plant science*, 7, 1419.
4. Ferentinos, K. P. (2018). Deep learning models for plant disease detection and diagnosis. *Computers and Electronics in Agriculture*, 145, 311-318.
5. Chollet, F. (2021). *Deep Learning with Python* (2nd ed.). Manning Publications.
