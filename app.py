import pathlib

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
)
import matplotlib.pyplot as plt
import joblib


class ReadmissionXGBModel:
    """
    Wrapper around the trained XGBoost model that:
    - remembers which features to use
    - knows which columns are categorical
    - applies the same integer encoding used during training
    """

    def __init__(self, model, feature_cols, cat_features, cat_maps):
        self.model = model
        self.feature_cols = feature_cols
        self.cat_features = cat_features
        self.cat_maps = cat_maps  # dict: col -> list of categories from TRAIN

    def _encode_categoricals(self, X: pd.DataFrame) -> pd.DataFrame:
        X_enc = X.copy()
        for col in self.cat_features:
            cats = self.cat_maps[col]
            cat_type = pd.api.types.CategoricalDtype(categories=cats)
            X_enc[col] = X_enc[col].astype(cat_type).cat.codes
            # Unseen categories become -1 (NaN code)
            X_enc[col] = X_enc[col].replace(-1, -1)
        return X_enc

    def predict_proba(self, X: pd.DataFrame):
        """
        X is a DataFrame with raw columns (age, n_spells, mean_los, pct_emerg,
        imd_quintile, sex, ethnicity_group, respiratory_group_mode)
        """
        X = X[self.feature_cols].copy()
        X_enc = self._encode_categoricals(X)
        return self.model.predict_proba(X_enc)


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
DATA_PATH = pathlib.Path('data/processed/patient_level.parquet')
# pipeline: preprocess + XGB
MODEL_PATH = pathlib.Path('models/readmission_xgb_pipeline.pkl')


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
@st.cache_data
def load_data(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df


@st.cache_resource
def load_model(path: pathlib.Path):
    """Load the trained pipeline (preprocessing + XGBoost model)."""
    model = joblib.load(path)
    return model


def add_risk_predictions(df: pd.DataFrame, model, feature_cols: list, target_col: str) -> pd.DataFrame:
    X = df[feature_cols].copy()
    y = df[target_col].astype(int)

    # Predict probabilities
    proba = model.predict_proba(X)[:, 1]
    df = df.copy()
    df["risk_score"] = proba
    df["y_true"] = y
    return df


def plot_pr_curve(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(recall, precision, label=f"Model (AP={ap:.3f})")
    ax.hlines(
        y_true.mean(),
        xmin=0,
        xmax=1,
        linestyle="--",
        label=f"Baseline (prevalence={y_true.mean():.3f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall curve")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_threshold_f1_curve(y_true, y_score):
    thresholds = np.linspace(0.01, 0.99, 99)
    f1s = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        f1s.append(f1_score(y_true, y_pred, zero_division=0))

    best_idx = int(np.argmax(f1s))
    best_thr = thresholds[best_idx]
    best_f1 = f1s[best_idx]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(thresholds, f1s)
    ax.axvline(best_thr, linestyle="--",
               label=f"Best F1={best_f1:.3f} at t={best_thr:.3f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1 score")
    ax.set_title("F1 vs threshold")
    ax.legend()
    fig.tight_layout()
    return fig, best_thr, best_f1


def plot_risk_by_group(df: pd.DataFrame, group_col: str, title: str):
    grouped = (
        df.groupby(group_col, observed=False)["risk_score"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(grouped[group_col].astype(str), grouped["risk_score"])
    ax.set_ylabel("Mean predicted risk")
    ax.set_xlabel(group_col.replace("_", " ").title())
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    return fig


def plot_feature_importance(model, feature_names: list):
    """Use model.feature_importances_ for a simple importance bar chart."""
    inner_model = getattr(model, "model", model)
    importances = getattr(inner_model, "feature_importances_", None)

    if importances is None:
        return None

    fi = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.barh(fi["feature"], fi["importance"], color="#1f77b4")

    ax.set_xlabel("Importance", fontsize=10)
    ax.set_title("Feature importance", fontsize=11, pad=6)

    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9)

    plt.tight_layout()
    return fig


# -------------------------------------------------------------------
# Streamlit app
# -------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Respiratory Readmission Risk Dashboard",
        layout="wide",
    )

    st.title("Respiratory Readmission Risk Dashboard")

    st.markdown(
        """
This dashboard summarises modelled 30-day readmission risk for respiratory admissions,
using a patient-level dataset derived from synthetic HES APC data and a trained XGBoost model.
        """
    )

    # -----------------------------
    # Load data + model
    # -----------------------------
    if not DATA_PATH.exists():
        st.error(
            f"Data file not found at {DATA_PATH}. Please generate patient_level.parquet first.")
        return

    if not MODEL_PATH.exists():
        st.error(
            f"Model file not found at {MODEL_PATH}.\n"
            "Save your trained pipeline (preprocessing + XGBoost) from the modelling notebook using joblib.dump."
        )
        return

    df_raw = load_data(DATA_PATH)
    pipeline = load_model(MODEL_PATH)

    # Define features used in the model
    feature_cols = [
        "age",
        "n_spells",
        "mean_los",
        "pct_emerg",
        "imd_quintile",
        "sex",
        "ethnicity_group",
        "respiratory_group_mode",
    ]
    target_col = "readmit_30d"

    missing_cols = [c for c in feature_cols +
                    [target_col] if c not in df_raw.columns]
    if missing_cols:
        st.error(
            f"The following required columns are missing from the data: {missing_cols}")
        return

    df = add_risk_predictions(df_raw, pipeline, feature_cols, target_col)

    y_true = df["y_true"].values
    y_score = df["risk_score"].values

    # -----------------------------
    # KPIs
    # -----------------------------
    total_patients = len(df)
    readmit_rate = y_true.mean()
    mean_pct_emerg = df["pct_emerg"].mean()
    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Patients", f"{total_patients:,}")
    kpi_cols[1].metric("Readmission rate", f"{100*readmit_rate:.2f}%")
    kpi_cols[2].metric("Mean emergency proportion",
                       f"{100*mean_pct_emerg:.1f}%")
    kpi_cols[3].metric("ROC-AUC", f"{roc_auc:.3f}")
    kpi_cols[4].metric("PR-AUC", f"{pr_auc:.3f}")

    st.markdown("---")

    # -----------------------------
    # Risk landscape (resp group / IMD)
    # -----------------------------
    st.subheader("Risk landscape by clinical and deprivation groups")

    col1, col2 = st.columns(2)

    with col1:
        fig_resp = plot_risk_by_group(
            df, "respiratory_group_mode", "Mean risk by respiratory group")
        st.pyplot(fig_resp)

        st.markdown("""These bars show the average predicted 30-day readmission risk for patients in each respiratory category. 
                    Asthma, COPD and Other respiratory groups have the highest model-estimated risk, whereas Pneumonia shows 
                    noticeably lower risk. These values reflect patterns learned by the model, not true causal effects.""")

    with col2:
        fig_imd = plot_risk_by_group(
            df, "imd_quintile", "Mean risk by IMD quintile")
        st.pyplot(fig_imd)

        st.markdown("""The deprivation gradient is relatively shallow. Average predicted risk varies only slightly across IMD quintiles, 
                    indicating that deprivation plays a smaller role in the model compared with clinical history 
                    (e.g., number of prior spells).""")

    st.markdown("---")

    # -----------------------------
    # Model diagnostics (PR curve + threshold-F1)
    # -----------------------------
    st.subheader("Model diagnostics")

    diag_col1, diag_col2 = st.columns(2)

    with diag_col1:
        fig_pr = plot_pr_curve(y_true, y_score)
        st.pyplot(fig_pr)

        st.markdown("""Because readmissions are rare (~2% prevalence), the PR curve is the most informative metric. 
                    The model’s average precision (~0.58) is substantially higher than the baseline (prevalence at 0.02), 
                    meaning the model is far better than chance at identifying high-risk patients.""")

    with diag_col2:
        fig_thr, best_thr, best_f1 = plot_threshold_f1_curve(y_true, y_score)
        st.pyplot(fig_thr)

        st.markdown("""This plot helps choose an operating threshold based on the recall–precision trade-off. 
                    The model achieves its best F1 score (~0.55) at a threshold of ~0.28, which provides a balanced compromise: 
                    good sensitivity while controlling false positives.""")

    st.markdown("---")

    # -----------------------------
    # Feature importance
    # -----------------------------
    st.subheader("Feature importance")

    # Try to access the underlying XGB model if the pipeline wraps it
    xgb_model = pipeline

    fig_fi = plot_feature_importance(xgb_model, feature_cols)

    if fig_fi is not None:
        st.pyplot(fig_fi, use_container_width=False)
    else:
        st.info("Feature importances not available on this model object.")

    st.markdown("""The number of prior spells overwhelmingly dominates feature importance, 
                followed by emergency admission percentage and mean length of stay. 
                Demographic variables such as age, sex, ethnicity and IMD contribute far less to model predictions.""")

    st.markdown("---")

    # -----------------------------
    # High-risk cohort explorer
    # -----------------------------
    st.subheader("High-risk cohort explorer")

    st.markdown("""This table lists patients whose predicted risk exceeds the chosen threshold. 
                It enables case-level review and supports operational planning (e.g., targeted follow-ups). 
                Sorting by risk score highlights the most clinically complex patients, 
                often with high spell counts and mixed respiratory histories.""")

    default_thr = float(np.round(best_thr, 3)
                        ) if "best_thr" in locals() else 0.2
    thr = st.slider(
        "Risk threshold for 'high risk'",
        min_value=0.0,
        max_value=1.0,
        value=default_thr,
        step=0.01,
    )

    df_high = df[df["risk_score"] >= thr].copy()
    df_high = df_high.sort_values("risk_score", ascending=False)

    st.write(
        f"Patients above threshold ({thr:.2f}): **{len(df_high):,}** "
        f"({len(df_high) / len(df):.2%} of all patients)"
    )

    df_disp = df_high.copy()
    df_disp = df_disp.rename(columns={
        "risk_score": "Risk score",
        "readmit_30d": "Readmitted 30 days",
        "age": "Age",
        "respiratory_group_mode": "Respiratory group",
        "n_spells": "Number of spells",
        "mean_los": "Mean LOS (days)",
        "pct_emerg": "% Emergency spells",
        "imd_quintile": "IMD quintile",
        "sex": "Sex",
        "ethnicity_group": "Ethnicity"
    })

    df_disp["Risk score"] = df_disp["Risk score"].round(3)
    df_disp["Age"] = df_disp["Age"].round(0)
    df_disp["Mean LOS (days)"] = df_disp["Mean LOS (days)"].round(2)

    # Convert to % string
    df_disp["% Emergency spells"] = (
        df_disp["% Emergency spells"] * 100).round(1).astype(str) + "%"

    # Show only selected columns in preferred order
    cols = [
        "Risk score",
        "Readmitted 30 days",
        "Age",
        "Respiratory group",
        "Number of spells",
        "Mean LOS (days)",
        "% Emergency spells",
        "IMD quintile",
        "Sex",
        "Ethnicity"
    ]

    st.dataframe(df_disp[cols].head(200), use_container_width=True)


if __name__ == "__main__":
    main()
