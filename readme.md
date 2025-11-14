Comparative Vision Study
This project conducts a comparative study of five deep learning models across three different tasks related to lung cancer analysis from CT scans.

Core Model Architectures
The following five models are evaluated in each of the three approaches:

DenseNet-121: A convolutional network that uses dense connections between layers to improve feature propagation.

EfficientNet-B0: A model that uniformly scales network depth, width, and resolution for high efficiency and accuracy.

ResNet-50: A classic residual network that uses skip connections to enable the training of very deep architectures.

MobileNet-V2: A lightweight, fast architecture designed for high performance on mobile and resource-constrained devices.

VGG-16: A foundational deep convolutional network known for its simple and effective 3x3 convolutional layer structure.

Comparative Approaches
M1: CT Multi-Class Classification
About this Approach: This task trains the models to perform 3-class classification on individual CT image slices. The goal is to categorize each slice as 'Normal', 'Benign', or 'Malignant'.

Dataset: This approach uses The IQ-OTH/NCCD Lung Cancer Dataset.

Link: https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset

M2: LIDC Nodule Segmentation
About this Approach: This task adapts the models to serve as encoders within a U-Net architecture to perform binary segmentation. The goal is to generate a pixel-level mask outlining the precise area of a lung nodule, separating it from the surrounding healthy tissue.

Dataset: This approach uses a custom-processed subset of the LIDC-IDRI (Lung Image Database Consortium) dataset.

Link: https://www.kaggle.com/datasets/zhangweiled/lidcidri

M3: Lung-RADS Binary Classification
About this Approach: This task trains the models to perform 2-class classification based on the Lung-RADS (Lung Cancer Screening Reporting and Data System) score. The goal is to classify slices as 'Negative' (Lung-RADS 1-2) or 'Positive' (Lung-RADS 3-4).

Dataset: This approach uses a public dataset that fuses LIDC-IDRI data with local Kazakhstani data, re-labeled by radiologists with Lung-RADS scores.

Link: https://data.mendeley.com/datasets/5rr22hgzwr/1