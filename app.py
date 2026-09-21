"""
Breast Cancer CAD Decision Support System — Streamlit Interface

Deserializes the fitted SVC pipeline, accepts clinical FNA biopsy inputs,
and returns diagnostic predictions with probability distributions.
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Breast Cancer CAD Decision Support System",
    page_icon="🔬",
    layout="wide",
)

MODEL_PATH = os.path.join("models", "best_cancer_pipeline.joblib")


@st.cache_resource
def load_pipeline():
    """Load the serialized sklearn pipeline from disk."""
    if not os.path.exists(MODEL_PATH):
        st.error(
            "Model file not found. Please run `python src/train.py` first."
        )
        st.stop()
    return joblib.load(MODEL_PATH)


pipeline = load_pipeline()

# ── Header & Metadata ────────────────────────────────────────────────────
st.title("🔬 Breast Cancer Diagnostic & Classification System")
st.markdown(
    """
**Computer-Aided Diagnostic (CAD) Decision Support Platform**
*Trained on the Wisconsin Diagnostic Breast Cancer (WDBC) Dataset*

**Model Architecture:** Support Vector Classifier (RBF Kernel)
with Z-Score Standardization
"""
)
st.write("---")

# ── Sidebar — Clinical Input Sliders ─────────────────────────────────────
st.sidebar.header("Biopsy Nuclear Metrics")
st.sidebar.markdown(
    "Configure patient cell nucleus measurements extracted from "
    "digitized FNA slides:"
)


def user_input_features() -> pd.DataFrame:
    """Collect primary mean metrics from the sidebar and extrapolate the rest."""
    radius_mean = st.sidebar.slider("Radius (Mean)", 6.0, 30.0, 14.12)
    texture_mean = st.sidebar.slider("Texture (Mean)", 9.0, 40.0, 19.28)
    perimeter_mean = st.sidebar.slider("Perimeter (Mean)", 43.0, 190.0, 91.96)
    area_mean = st.sidebar.slider("Area (Mean)", 140.0, 2500.0, 654.88)
    smoothness_mean = st.sidebar.slider("Smoothness (Mean)", 0.05, 0.20, 0.096)
    compactness_mean = st.sidebar.slider(
        "Compactness (Mean)", 0.02, 0.35, 0.104
    )
    concavity_mean = st.sidebar.slider("Concavity (Mean)", 0.0, 0.45, 0.088)
    concave_points_mean = st.sidebar.slider(
        "Concave Points (Mean)", 0.0, 0.20, 0.048
    )

    # Secondary (SE) and Worst metrics populated with representative baselines
    data = {
        "radius_mean": radius_mean,
        "texture_mean": texture_mean,
        "perimeter_mean": perimeter_mean,
        "area_mean": area_mean,
        "smoothness_mean": smoothness_mean,
        "compactness_mean": compactness_mean,
        "concavity_mean": concavity_mean,
        "concave points_mean": concave_points_mean,
        "symmetry_mean": 0.181,
        "fractal_dimension_mean": 0.062,
        "radius_se": 0.405,
        "texture_se": 1.216,
        "perimeter_se": 2.866,
        "area_se": 40.33,
        "smoothness_se": 0.007,
        "compactness_se": 0.025,
        "concavity_se": 0.031,
        "concave points_se": 0.011,
        "symmetry_se": 0.020,
        "fractal_dimension_se": 0.003,
        "radius_worst": radius_mean * 1.15,
        "texture_worst": texture_mean * 1.2,
        "perimeter_worst": perimeter_mean * 1.15,
        "area_worst": area_mean * 1.25,
        "smoothness_worst": smoothness_mean * 1.1,
        "compactness_worst": compactness_mean * 1.15,
        "concavity_worst": concavity_mean * 1.2,
        "concave points_worst": concave_points_mean * 1.15,
        "symmetry_worst": 0.290,
        "fractal_dimension_worst": 0.083,
    }
    return pd.DataFrame(data, index=[0])


input_df = user_input_features()

# ── Main Interface Layout ────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Submitted Patient Morphometric Profile")
    st.dataframe(
        input_df.iloc[:, :8].style.format("{:.3f}"), use_container_width=True
    )
    st.caption(
        "Displaying primary mean parameters. Standard error and extreme "
        "boundary factors are dynamically extrapolated."
    )

if st.button("Run Diagnostic Evaluation", type="primary"):
    prediction = pipeline.predict(input_df)[0]
    prediction_proba = pipeline.predict_proba(input_df)[0]

    with col2:
        st.subheader("Diagnostic Assessment")

        if prediction == 1:
            st.error("⚠️ **MALIGNANT NEOPLASM DETECTED**")
            malignant_prob = prediction_proba[1] * 100
            st.metric(
                label="Malignancy Confidence", value=f"{malignant_prob:.2f}%"
            )
            st.progress(prediction_proba[1])
            st.warning(
                "**Clinical Protocol:** High probability of invasive tissue. "
                "Recommend immediate secondary histological biopsy."
            )
        else:
            st.success("✅ **BENIGN TISSUE CLASSIFICATION**")
            benign_prob = prediction_proba[0] * 100
            st.metric(
                label="Benign Confidence", value=f"{benign_prob:.2f}%"
            )
            st.progress(prediction_proba[0])
            st.info(
                "**Clinical Protocol:** Morphological attributes conform to "
                "standard non-invasive tissue structures."
            )

st.write("---")
st.caption(
    "Academic Decision Support Prototype | Developed for Summer Industrial "
    "Training Assessment (2026)"
)
