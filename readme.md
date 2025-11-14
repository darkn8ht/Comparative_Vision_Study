# Comparative Vision Study

This project conducts a **comparative study of five deep learning models** across **three different tasks** related to lung cancer analysis using CT scans.

---

## Core Model Architectures

The following five models are evaluated across all three approaches:

- **DenseNet-121**: A convolutional network that uses dense connections between layers to improve feature propagation.  
- **EfficientNet-B0**: A model that uniformly scales network depth, width, and resolution for high efficiency and accuracy.  
- **ResNet-50**: A classic residual network that uses skip connections to enable the training of very deep architectures.  
- **MobileNet-V2**: A lightweight, fast architecture designed for high performance on mobile and resource-constrained devices.  
- **VGG-16**: A foundational deep convolutional network known for its simple and effective 3×3 convolutional layer structure.

---

## Comparative Approaches

### **M1: CT Multi-Class Classification**

**About this Approach:**  
This task trains the models to perform **3-class classification** on individual CT image slices.  
The goal is to categorize each slice as `Normal`, `Benign`, or `Malignant`.

**Dataset:**  
- **Name:** The IQ-OTH/NCCD Lung Cancer Dataset  
- **Link:** [https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset](https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset)

---

### **M2: LIDC Nodule Segmentation**

**About this Approach:**  
In this task, models are used as **encoders within a U-Net architecture** for **binary segmentation**.  
The goal is to generate a pixel-level mask outlining the precise area of a lung nodule, separating it from surrounding healthy tissue.

**Dataset:**  
- **Name:** Custom-processed subset of the LIDC-IDRI (Lung Image Database Consortium) dataset  
- **Link:** [https://www.kaggle.com/datasets/zhangweiled/lidcidri](https://www.kaggle.com/datasets/zhangweiled/lidcidri)

---

### **M3: Lung-RADS Binary Classification**

**About this Approach:**  
This task involves **2-class classification** based on the **Lung-RADS (Lung Cancer Screening Reporting and Data System)** score.  
The goal is to classify slices as:
- `Negative` (Lung-RADS 1–2)  
- `Positive` (Lung-RADS 3–4)

**Dataset:**  
- **Name:** Lung-RADS-labeled dataset combining LIDC-IDRI and local Kazakhstani data  
- **Link:** [https://data.mendeley.com/datasets/5rr22hgzwr/1](https://data.mendeley.com/datasets/5rr22hgzwr/1)

---

## Summary Overview

| Approach | Task Type | Classes | Dataset | Description |
|-----------|------------|----------|-----------|--------------|
| **M1** | Multi-class Classification | 3 (`Normal`, `Benign`, `Malignant`) | IQ-OTH/NCCD | Slice-level classification |
| **M2** | Segmentation | 2 (Nodule / Background) | LIDC-IDRI | Pixel-level mask generation |
| **M3** | Binary Classification | 2 (`Negative`, `Positive`) | Lung-RADS-labeled data | Screening-level classification |

---

## Purpose

This comparative study aims to evaluate each model’s **performance**, **generalization ability**, and **architectural efficiency** across distinct computer vision tasks in the domain of **lung cancer detection and analysis**.
