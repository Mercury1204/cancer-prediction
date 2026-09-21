# Breast Cancer Diagnostic Prediction and Classification Using Supervised Machine Learning

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-v1.3-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)

An end-to-end machine learning classification pipeline developed to assist in Computer-Aided Diagnostics (CAD) by predicting tumor malignancy from Fine Needle Aspirate (FNA) cell nucleus metrics. Developed as part of the **Summer Industrial Training Program (2026)** under the supervision of **Dr. Alok Yadav (YBI Foundation)** and submitted to the **Department of Electronics and Communication Engineering, Jaypee Institute of Information Technology, Noida**.

---

## 📌 Project Overview

Diagnostic errors in computational oncology are asymmetric: a **False Negative (Type II error)** carries lethal clinical risks by leaving malignant neoplasms undetected. This project implements a leak-free data preprocessing pipeline, benchmarks multiple machine learning architectures, and tunes hyperparameters targeting maximum **Recall (Sensitivity)** while preserving overall accuracy.

### Key Deliverables

* **Data Hygiene & Preprocessing:** Stripped identifier markers, encoded binary classes, and implemented Z-score standardization strictly within cross-validation folds.
* **Algorithmic Benchmarking:** Evaluated Logistic Regression, Support Vector Classifier (RBF), Decision Trees, and Random Forests.
* **Clinical Optimization:** Executed Stratified 5-Fold GridSearchCV prioritizing Recall.
* **Interactive Deployment:** Built a real-time clinical decision-support interface using Streamlit.

---

## 📊 Benchmark Results

Evaluated across a 20% holdout test partition (114 samples: 71 Benign, 43 Malignant):

| Architecture | Accuracy | Precision | Recall (Sensitivity) | F1-Score | Testing ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (L2)** | 0.9737 | 0.9762 | 0.9535 | 0.9647 | 0.9941 |
| **Support Vector Classifier (RBF)** | **0.9825** | **0.9773** | **0.9767** | **0.9770** | **0.9967** |
| **Decision Tree (Pruned)** | 0.9298 | 0.9091 | 0.9070 | 0.9080 | 0.9256 |
| **Random Forest Ensemble** | 0.9649 | 0.9756 | 0.9302 | 0.9524 | 0.9918 |

> **Clinical Finding:** The optimized Support Vector Classifier achieved **97.67% Recall**, successfully limiting Type II errors to a single instance on the holdout test set with zero False Positives.

---

## 🗂 Repository Structure

```
cancer-prediction/
├── data/
│   └── Cancer.csv                      # Wisconsin Diagnostic Breast Cancer dataset
├── models/
│   └── best_cancer_pipeline.joblib     # Serialized StandardScaler + Best Model
├── notebooks/
│   └── exploratory_data_analysis.ipynb # Visualizations, correlation heatmaps, pairplots
├── src/
│   ├── __init__.py
│   ├── data_loader.py                  # Ingestion, cleaning, and train-test splitting
│   ├── evaluate.py                     # Confusion matrices, ROC-AUC, classification reports
│   └── train.py                        # Multi-model benchmarking and GridSearchCV tuning
├── app.py                              # Streamlit clinical diagnostic web interface
├── requirements.txt                    # Project runtime dependencies
└── README.md                           # Formal documentation
```

---

## 🚀 Installation & Local Execution

### 1. Clone the Repository

```bash
git clone https://github.com/Mercury1204/cancer-prediction.git
cd cancer-prediction
```

### 2. Configure Virtual Environment & Dependencies

```bash
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Execute Model Training Pipeline

```bash
python src/train.py
```

This fetches the dataset, runs the cross-validated benchmarks, and persists `best_cancer_pipeline.joblib` inside the `models/` directory.

### 4. Launch the Streamlit Diagnostic Web App

```bash
streamlit run app.py
```

---

## 🗂 Dataset Attribution

The dataset utilized is the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset sourced via the [YBI Foundation Dataset Repository](https://github.com/YBIFoundation/Dataset/raw/main/Cancer.csv).

* **Records:** 569 patient instances (357 Benign, 212 Malignant)
* **Predictive Attributes:** 30 continuous morphological cell nucleus features

---

## 📄 License

This project was developed for academic purposes as part of the Summer Industrial Training Program (2026).
