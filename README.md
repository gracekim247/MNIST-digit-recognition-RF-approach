# Handwritten Digit Recognition on MNIST using Random Forest

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gracekim247/MNIST-digit-recognition-RF-approach/blob/main/src.ipynb)
[![Project Report](https://img.shields.io/badge/PDF-Project_Report-red)](https://drive.google.com/file/d/1aeroZVgf0BEDWXPz9j6-5HVCHsOo9dFa/view?usp=drive_link)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Project Overview

This project implements a **Random Forest Classifier** to recognize handwritten digits from the famous **MNIST dataset**. Unlike modern approaches that rely heavily on Convolutional Neural Networks (CNNs), this project demonstrates the power of classical ensemble learning. By utilizing raw pixel intensities and rigorous cross-validation, the model achieves a competitive accuracy of **~97%**, maintaining high interpretability and efficiency.

**Read the full technical report here:** [Final Project Report (PDF)](https://drive.google.com/file/d/1aeroZVgf0BEDWXPz9j6-5HVCHsOo9dFa/view?usp=drive_link)

This work was conducted as a final term project for the CSCI367 Digital Image Processing course.

## Key Features

* **Algorithm:** Random Forest Ensemble (100 Decision Trees).
* **Preprocessing:** Manual flattening of 28x28 images to 784-feature vectors with [0,1] normalization.
* **Validation:** 10-fold cross-validation to ensure model robustness (96.76% mean validation accuracy).
* **Evaluation:** Comprehensive analysis using F1-scores, Confusion Matrices, and Per-class Accuracy.
* **No Black Boxes:** Does not use CNNs or automated feature extraction layers.

## Results

The model was trained on 60,000 images and evaluated on a held-out test set of 10,000 images.

| Metric | Score |
| :--- | :--- |
| **Overall Accuracy** | **97.04%** |
| **Validation Accuracy (Mean)** | 96.76% ($\sigma = 0.004$) |
| **Macro Average F1-Score** | 0.97 |

<img width="440" height="436" alt="Image" src="https://github.com/user-attachments/assets/052640fe-2778-48cf-a208-f1038f023c7f" />

### Error Analysis
While performance is high, the model exhibits minor confusion on topologically similar digits, specifically:
* **4 vs 9:** Due to the closed top loop similarity.
* **7 vs 2:** Due to upper stroke similarities.

<img width="452" height="412" alt="Image" src="https://github.com/user-attachments/assets/61c95721-1560-42ba-b1a5-6c450b3c64b1" />

## Technologies Used

* **Python 3**
* **Scikit-Learn** (Random Forest, Metrics, Model Selection)
* **TensorFlow/Keras** (Used solely for simplified Data Loading)
* **Pandas & NumPy** (Data Manipulation)
* **Matplotlib** (Visualization)

## How to Run - Google Colab

Click the badge at the top of this README or [click here](https://colab.research.google.com/github/gracekim247/MNIST-digit-recognition-RF-approach/blob/main/src.ipynb) to run the notebook directly in your browser.

## Authors & Contributions

* **Grace Kim**: Data preprocessing pipeline, normalization, 10-fold cross-validation implementation, and confusion matrix visualization.
* **Moazam Hussain**: Random Forest model development, hyperparameter tuning, error analysis, and per-class metric calculations.

## License

This project is open-source and available under the [MIT License](LICENSE).
