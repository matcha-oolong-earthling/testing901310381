"""
SepsisAlert — SMART on FHIR Sepsis Early Warning System
Streamlit web application for Team J-VISA.
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import RocCurveDisplay, confusion_matrix
from sklearn.model_selection import train_test_split

from jvisa.csv_mapper import FHIRToDataFrameMapper
from jvisa.model import SepsisRandomForest

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = pathlib.Path("dataset/MIMIC-IV-ICU-synthetic")
NDJSON_PATH = DATA_DIR / "bundles.ndjson"
JSON_PATH = DATA_DIR / "bundles.json"

st.set_page_config(
    page_title="SepsisAlert | J-VISA",
    page_icon=":hospital:",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_fhir_data(path: str, impute_strategy: str) -> pd.DataFrame:
    mapper = FHIRToDataFrameMapper()
    p = pathlib.Path(path)
    df = mapper.from_ndjson(p) if p.suffix == ".ndjson" else mapper.from_json(p)
    df = mapper.impute(df, strategy=impute_strategy)
    return df


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title(":hospital: SepsisAlert")
st.markdown(
    "**AI-powered sepsis early warning system** built on SMART on FHIR. "
    "This app demonstrates the full pipeline: "
    "FHIR Bundles :arrow_right: Tabular Data :arrow_right: Random Forest Prediction."
)
st.divider()

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Settings")

data_source = st.sidebar.radio(
    "Data source",
    ["bundles.ndjson", "bundles.json"],
    help="Select the FHIR bundle file format.",
)
data_path = str(DATA_DIR / data_source)

impute_strategy = st.sidebar.selectbox(
    "Imputation strategy",
    ["median", "mean", "zero"],
    help="How to fill missing values in numeric columns.",
)

st.sidebar.divider()
st.sidebar.subheader("Model hyperparameters")

n_estimators = st.sidebar.slider("Number of trees", 50, 500, 200, step=50)
max_depth = st.sidebar.slider("Max tree depth (0 = unlimited)", 0, 30, 0)
test_size = st.sidebar.slider("Test set fraction", 0.1, 0.4, 0.2, step=0.05)
noise_scale = st.sidebar.slider(
    "Noise scale",
    0.0, 4.0, 0.0, step=0.5,
    help="Gaussian noise injected per feature (x feature std). "
         "Use > 0 to simulate real-world noise on synthetic data.",
)

max_depth_param = None if max_depth == 0 else max_depth

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = load_fhir_data(data_path, impute_strategy)

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------
tab_overview, tab_data, tab_model, tab_predict = st.tabs(
    ["Overview", "Data Explorer", "Model Results", "Patient Lookup"]
)

# ===== TAB 1: Overview =====================================================
with tab_overview:
    col1, col2, col3, col4 = st.columns(4)
    n_sepsis = int(df["sepsis_label"].sum())
    n_total = len(df)
    n_missing = int(df.select_dtypes(include="number").isna().sum().sum())

    col1.metric("Total Encounters", f"{n_total:,}")
    col2.metric("Sepsis Cases", f"{n_sepsis:,}")
    col3.metric("Prevalence", f"{n_sepsis / n_total:.1%}")
    col4.metric("Missing Values (post-impute)", f"{n_missing:,}")

    st.subheader("Pipeline Architecture")
    st.code(
        "CSV (MIMIC-IV-ICU)\n"
        "  |  jvisa.fhir_mapper\n"
        "  v\n"
        "FHIR R4 Bundles (bundles.ndjson / bundles.json)\n"
        "  |  jvisa.csv_mapper\n"
        "  v\n"
        "pandas DataFrame (imputed)\n"
        "  |  jvisa.model (Random Forest)\n"
        "  v\n"
        "Sepsis Risk Prediction",
        language=None,
    )

    st.subheader("Feature Categories")
    feat_table = pd.DataFrame(
        {
            "Category": [
                "Vital Signs",
                "Laboratory Results",
                "Clinical Scores",
                "Body Measurements",
                "Comorbidities",
                "Interventions",
                "Demographics / Encounter",
            ],
            "Examples": [
                "HR, SBP, DBP, MAP, SpO2, Temperature, Respiratory Rate (mean/max/min)",
                "WBC, Lactate, Creatinine, Platelets, Bilirubin, Glucose, pH, INR, Na/K/Cl/HCO3",
                "SOFA, APACHE IV, qSOFA, SIRS, GCS, FiO2, Sedation",
                "Weight, Height, BMI",
                "Diabetes, Hypertension, CHF, COPD, CKD, Liver Disease, A-Fib, Cancer, etc.",
                "Vasopressors, Mechanical Ventilation, Antibiotics, Insulin, IV Fluids",
                "Age, Gender, Admit Source, LOS, Readmission Flag",
            ],
        }
    )
    st.table(feat_table)


# ===== TAB 2: Data Explorer =================================================
with tab_data:
    st.subheader("Parsed DataFrame")
    st.caption(f"{len(df)} rows x {len(df.columns)} columns (after FHIR reverse mapping)")
    st.dataframe(df.head(100), use_container_width=True, height=400)

    st.subheader("Summary Statistics")
    st.dataframe(
        df.describe().T.style.format("{:.2f}"),
        use_container_width=True,
        height=400,
    )

    st.subheader("Sepsis vs. Non-Sepsis Distribution")
    compare_col = st.selectbox(
        "Select a feature to compare",
        [c for c in df.select_dtypes(include="number").columns if c != "sepsis_label"],
        index=0,
    )
    fig_dist, ax_dist = plt.subplots(figsize=(8, 3.5))
    for label, color in [(0, "#4C8BF5"), (1, "#EA4335")]:
        subset = df.loc[df["sepsis_label"] == label, compare_col].dropna()
        ax_dist.hist(subset, bins=40, alpha=0.6, label=f"{'No Sepsis' if label == 0 else 'Sepsis'}", color=color)
    ax_dist.set_xlabel(compare_col)
    ax_dist.set_ylabel("Count")
    ax_dist.legend()
    ax_dist.set_title(f"Distribution of {compare_col} by Sepsis Label")
    st.pyplot(fig_dist)
    plt.close(fig_dist)


# ===== TAB 3: Model Results =================================================
with tab_model:
    st.subheader("Train & Evaluate")

    if st.button("Run Random Forest", type="primary"):
        with st.spinner("Training model ..."):
            model = SepsisRandomForest(
                n_estimators=n_estimators,
                max_depth=max_depth_param,
                noise_scale=noise_scale,
                test_size=test_size,
            )
            results = model.train_and_evaluate(df)

            # Store in session for the predict tab
            st.session_state["model"] = model
            st.session_state["results"] = results

    if "results" not in st.session_state:
        st.info("Click **Run Random Forest** to train the model and see results.")
    else:
        results = st.session_state["results"]
        model = st.session_state["model"]

        # Metrics cards
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{results.accuracy:.4f}")
        m2.metric("Precision", f"{results.precision:.4f}")
        m3.metric("Recall", f"{results.recall:.4f}")
        m4.metric("F1 Score", f"{results.f1:.4f}")
        m5.metric("ROC AUC", f"{results.roc_auc:.4f}" if results.roc_auc else "N/A")

        # Plots
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("**Confusion Matrix**")
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            cm = results.confusion_matrix
            im = ax_cm.imshow(cm, cmap="Blues")
            ax_cm.set_xticks([0, 1])
            ax_cm.set_yticks([0, 1])
            ax_cm.set_xticklabels(["No Sepsis", "Sepsis"])
            ax_cm.set_yticklabels(["No Sepsis", "Sepsis"])
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            for i in range(2):
                for j in range(2):
                    color = "white" if cm[i, j] > cm.max() / 2 else "black"
                    ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=18, color=color)
            fig_cm.colorbar(im, ax=ax_cm, fraction=0.046)
            st.pyplot(fig_cm)
            plt.close(fig_cm)

        with col_right:
            st.markdown("**Top 15 Feature Importances**")
            sorted_imp = sorted(
                results.feature_importances.items(), key=lambda x: x[1], reverse=True
            )[:15]
            fig_fi, ax_fi = plt.subplots(figsize=(5, 4))
            names = [n for n, _ in reversed(sorted_imp)]
            vals = [v for _, v in reversed(sorted_imp)]
            ax_fi.barh(names, vals, color="#4C8BF5")
            ax_fi.set_xlabel("Importance")
            ax_fi.tick_params(axis="y", labelsize=8)
            st.pyplot(fig_fi)
            plt.close(fig_fi)

        # ROC curve
        st.markdown("**ROC Curve**")
        # Re-derive predictions for ROC plot
        X, y = model._prepare(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=model.test_size, random_state=model.random_state, stratify=y,
        )
        y_proba = model.clf.predict_proba(X_test)[:, 1]
        fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
        RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax_roc, name="Random Forest")
        ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.3)
        st.pyplot(fig_roc)
        plt.close(fig_roc)

        # Classification report
        with st.expander("Full Classification Report"):
            st.code(results.classification_report)


# ===== TAB 4: Patient Lookup ================================================
with tab_predict:
    st.subheader("Individual Patient Sepsis Risk")

    if "model" not in st.session_state:
        st.info("Train the model first in the **Model Results** tab.")
    else:
        model = st.session_state["model"]
        patient_ids = df["subject_id"].astype(str).tolist()
        selected_id = st.selectbox("Select patient (subject_id)", patient_ids)

        patient_row = df[df["subject_id"].astype(str) == selected_id]
        if not patient_row.empty:
            proba = model.predict_proba(patient_row)[0]
            sepsis_prob = proba[1] if len(proba) == 2 else proba[0]
            actual = int(patient_row["sepsis_label"].iloc[0])

            col_a, col_b = st.columns(2)
            col_a.metric("Sepsis Probability", f"{sepsis_prob:.1%}")
            col_b.metric("Actual Label", "SEPSIS" if actual == 1 else "No Sepsis")

            if sepsis_prob >= 0.5:
                st.error(f"HIGH RISK — Model predicts sepsis (probability {sepsis_prob:.1%})")
            elif sepsis_prob >= 0.2:
                st.warning(f"MODERATE RISK — Elevated sepsis probability ({sepsis_prob:.1%})")
            else:
                st.success(f"LOW RISK — Sepsis probability {sepsis_prob:.1%}")

            st.markdown("**Patient Features**")
            # Show key features in a readable format
            display_row = patient_row.iloc[0].dropna()
            st.dataframe(
                pd.DataFrame({"Feature": display_row.index, "Value": display_row.values}),
                use_container_width=True,
                hide_index=True,
            )
