# CropDoctor - Plant Disease Classification Model

High-accuracy convolutional neural network for plant disease identification from leaf images.

## Project Overview

This repository contains a custom-trained deep learning model for automatic detection and classification of plant diseases from leaf photographs.

The model was trained from scratch (transfer learning) on a large combined dataset of plant leaf images and achieves strong performance suitable for real-world agricultural applications.

## Model Details

- Architecture: EfficientNet-B3 (fine-tuned)
- Framework: PyTorch + timm
- Training augmentations: Albumentations (RandomResizedCrop, flips, rotations, brightness/contrast, CoarseDropout, etc.)
- Optimizer: AdamW + CosineAnnealingLR scheduler
- Regularization techniques: MixUp, Test-Time Augmentation (TTA), label smoothing
- Input size: 300×300 (training), 380×380 possible for inference with TTA
- Final validation accuracy: 99.4%
- Number of classes: ~60 (covering major crops: Tomato, Potato, Apple, Grape, Corn, Rice, Pepper, etc.)
- Inference latency: 300–600 ms on CPU (ONNX optimized)
- Model format: ONNX Runtime (exported for fast cross-platform inference)

## Dataset

- Primary source: New Plant Diseases Dataset (87k+ images)
- Additional source: Original PlantVillage dataset (~54k images)
- Total training images: ~120k–140k (after combining and cleaning)
- Severe class imbalance addressed through weighted sampling and augmentations

Class distribution is highly skewed (some diseases have 10k+ samples, others <200).

## Performance Highlights

| Metric                  | Value          | Notes                              |
|-------------------------|----------------|------------------------------------|
| Validation Accuracy     | 99.1%          | After 15 epochs + TTA             |
| Top-3 Accuracy          | 99.9%          | Very high confidence in top predictions |
| Best single-class accuracy | >99.8%      | Many classes reach near-perfect   |
| Worst-class accuracy    | ~96–97%        | Rare/minority classes             |
| Model size (ONNX)       | ~20 MB         | Efficient for edge deployment     |

<img width="1165" height="876" alt="image" src="https://github.com/user-attachments/assets/5348588e-0567-4076-bf23-411e3417885f" />
<img width="1917" height="957" alt="image" src="https://github.com/user-attachments/assets/861f853e-e13b-4e0d-a540-dfd29fad6bbd" />

