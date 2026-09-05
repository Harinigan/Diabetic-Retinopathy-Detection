# Hybrid Image Enhancement with Deep Learning for Diabetic Retinopathy Detection

## Overview

This project presents a deep learning framework for automated **Diabetic
Retinopathy (DR)** detection using retinal fundus images. It combines
advanced image enhancement techniques with an ensemble of
**EfficientNet-B2** and **EfficientNet-B3** models to improve
classification performance across five DR stages.

## Features

-   Automated diabetic retinopathy detection
-   Hybrid image enhancement pipeline
-   EfficientNet-B2 & EfficientNet-B3 ensemble
-   Test-Time Augmentation (TTA)
-   Soft voting ensemble prediction
-   Flask-based web application
-   Evaluation using Accuracy, Precision, Recall, F1-Score, ROC-AUC, and
    Confusion Matrix

## Disease Classes

-   No_DR
-   Mild
-   Moderate
-   Severe
-   Proliferative DR

## Tech Stack

-   Python
-   PyTorch
-   OpenCV
-   NumPy
-   Scikit-learn
-   Flask
-   Matplotlib
-   EfficientNet-B2
-   EfficientNet-B3

## Project Structure

``` text
Dataset/
Processed_Dataset/
Split/
├── train/
├── val/
└── test/
Models/
├── dr_model_b2.pth
└── best_dr_b3.pth
preprocessing.py
split_dataset.py
train_eff_b2.py
train_eff_b3.py
evaluate_ensemble.py
app.py
requirements.txt
README.md
```

## Workflow

1.  Load retinal fundus images.
2.  Crop retinal region.
3.  Apply circular masking.
4.  Perform CLAHE enhancement.
5.  Resize and normalize images.
6.  Split dataset.
7.  Train EfficientNet-B2 and EfficientNet-B3.
8.  Perform ensemble prediction using TTA and soft voting.
9.  Evaluate the model.

## Installation

``` bash
git clone https://github.com/your-username/diabetic-retinopathy-detection.git
cd diabetic-retinopathy-detection
pip install -r requirements.txt
```

## Run

``` bash
python train_eff_b2.py
python train_eff_b3.py
python evaluate_ensemble.py
```

## Applications

-   AI-assisted ophthalmology
-   Automated DR screening
-   Medical image analysis
-   Smart healthcare systems

## Future Enhancements

-   Vision Transformer integration
-   Explainable AI (Grad-CAM)
-   Cloud deployment
-   Mobile application
-   Multi-disease retinal diagnosis

## Author

**Harini G**

MCA Student \| AI & Machine Learning Enthusiast

## License

This project is intended for educational and research purposes.
