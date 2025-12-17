import pathlib

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


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
DATA_PATH = pathlib.Path("data/processed/patient_level.parquet")
# pipeline: preprocess + XGB
MODEL_PATH = pathlib.Path("models/readmission_xgb_pipeline.pkl")


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
@st.cache_data
def load_data(path: pathlib.Path) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_resource
def load_model(path: pathlib.Path):
    """Load the trained pipeline (preprocessing + XGBoost model)."""
    return joblib.load(path)


def add_risk_predictions(
    df: pd.DataFrame, model, feature_cols: list, target_col: str
) -> pd.DataFrame:
    X = df[feature_cols].copy()
    y = df[target_col].astype(int)

    # Predict probabilities
    proba = model.predict_proba(X)[:, 1]
    out = df.copy()
    out["risk_score"] = proba
    out["y_true"] = y
    return out


def compute_best_f1_threshold(y_true, y_score):
    thresholds = np.linspace(0.01, 0.99, 99)
    f1s = []

    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        f1s.append(f1_score(y_true, y_pred, zero_division=0))

    f1s = np.array(f1s)
    best_idx = int(np.argmax(f1s))
    return thresholds[best_idx], f1s[best_idx]


def plot_pr_curve(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    prevalence = float(np.mean(y_true))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name=f"Model (AP={ap:.3f})",
            hovertemplate="Recall=%{x:.3f}<br>Precision=%{y:.3f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[prevalence, prevalence],
            mode="lines",
            name=f"Baseline (prevalence={prevalence:.3f})",
            line=dict(dash="dash"),
            hovertemplate="Baseline precision=%{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Precision–Recall curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(x=0.02, y=0.13, bgcolor="rgba(255,255,255,0.7)"),
    )

    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return fig


def plot_f1_vs_threshold(y_true, y_score):
    # NOTE: This recomputes the F1 curve for plotting. The best threshold
    # is computed once in main() for reuse elsewhere.
    thresholds = np.linspace(0.01, 0.99, 99)
    f1s = []

    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        f1s.append(f1_score(y_true, y_pred, zero_division=0))

    f1s = np.array(f1s)
    best_idx = int(np.argmax(f1s))
    best_thr = thresholds[best_idx]
    best_f1 = f1s[best_idx]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=f1s,
            mode="lines",
            name="F1 score",
            hovertemplate="Threshold=%{x:.2f}<br>F1=%{y:.3f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[best_thr],
            y=[best_f1],
            mode="markers",
            marker=dict(size=12),
            name=f"Best F1 ({best_f1:.3f})",
            hovertemplate="Best threshold=%{x:.2f}<br>F1=%{y:.3f}<extra></extra>",
        )
    )

    fig.add_vline(
        x=best_thr,
        line_dash="dash",
        annotation_text=f"Best threshold = {best_thr:.2f}",
        annotation_position="top right",
    )

    fig.update_layout(
        title="F1 score vs classification threshold",
        xaxis_title="Threshold",
        yaxis_title="F1 score",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(x=0.02, y=0.13, bgcolor="rgba(255,255,255,0.7)"),
    )

    return fig


def plot_risk_by_group(df: pd.DataFrame, group_col: str, title: str):
    grouped = (
        df.groupby(group_col, observed=False)["risk_score"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig = go.Figure()

    group_label = group_col.replace("_", " ").title()

    fig.add_trace(
        go.Bar(
            x=grouped[group_col].astype(str),
            y=grouped["risk_score"],
            hovertemplate=(
                f"{group_label}: %{{x}}<br>"
                "Mean predicted risk=%{y:.3f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=group_label,
        yaxis_title="Mean predicted risk",
        height=350,
        margin=dict(l=40, r=20, t=50, b=80),
    )
    return fig


def plot_feature_importance(model, feature_names: list):
    """Use model.feature_importances_ for a simple importance bar chart."""
    inner_model = getattr(model, "model", model)
    importances = getattr(inner_model, "feature_importances_", None)

    if importances is None:
        return None

    fi = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
    )

    # Dynamic sizing so labels don’t squash
    n = len(fi)
    height = max(350, min(900, 18 * n + 120))
    left_margin = 220  # increase if your feature names are very long

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=fi["importance"],
            y=fi["feature"].astype(str),
            orientation="h",
            hovertemplate="Feature: %{y}<br>Importance=%{x:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Feature importance",
        xaxis_title="Importance",
        yaxis_title="",
        height=height,
        margin=dict(l=left_margin, r=20, t=50, b=40),
        showlegend=False,
    )

    # Most important at the top
    fig.update_yaxes(autorange="reversed")

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
            f"Data file not found at {DATA_PATH}. Please generate patient_level.parquet first."
        )
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

    # Compute once for reuse across the dashboard
    best_thr, best_f1 = compute_best_f1_threshold(y_true, y_score)

    # -----------------------------
    # KPIs
    # -----------------------------
    total_patients = len(df)
    readmit_rate = float(y_true.mean())
    mean_pct_emerg = float(df["pct_emerg"].mean())
    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    kpi_cols = st.columns(5)

    kpi_cols[0].metric(
        "Patients",
        f"{total_patients:,}",
        help="Number of patients in the evaluation cohort.",
    )

    kpi_cols[1].metric(
        "Readmission rate",
        f"{100 * readmit_rate:.2f}%",
        help="Observed prevalence of 30-day readmission in the data.",
    )

    kpi_cols[2].metric(
        "Mean emergency proportion",
        f"{100 * mean_pct_emerg:.1f}%",
        help="Average proportion of admissions that were emergencies.",
    )

    kpi_cols[3].metric(
        "ROC-AUC",
        f"{roc_auc:.3f}",
        help="Overall ranking ability across thresholds; can appear optimistic with rare outcomes.",
    )

    kpi_cols[4].metric(
        "PR-AUC (AP)",
        f"{pr_auc:.3f}",
        help=f"Precision–recall summary; baseline equals prevalence (~{readmit_rate:.2%}).",
    )

    st.markdown("---")

    # -----------------------------
    # Risk landscape (resp group / IMD)
    # -----------------------------
    st.subheader("Risk landscape by clinical and deprivation groups")

    col1, col2 = st.columns(2)

    with col1:
        fig_resp = plot_risk_by_group(
            df, "respiratory_group_mode", "Average predicted risk by respiratory group"
        )
        fig_resp.update_layout(xaxis_title="Respiratory group")
        st.plotly_chart(fig_resp, use_container_width=True)

        st.markdown("""These bars show the average predicted 30-day readmission risk for patients in each respiratory category.
Asthma, COPD, and other respiratory conditions have the highest model-estimated risk, while pneumonia is 
associated with a noticeably lower predicted risk. These differences reflect patterns learned by the model 
from the data and should not be interpreted as causal effects.""")

    with col2:
        fig_imd = plot_risk_by_group(
            df, "imd_quintile", "Average predicted risk by IMD quintile")
        fig_imd.update_layout(xaxis_title="IMD quintile")
        st.plotly_chart(fig_imd, use_container_width=True)

        st.markdown("""The deprivation gradient is relatively shallow. Average predicted risk varies only modestly across 
IMD quintiles, suggesting that socioeconomic deprivation contributes less to the model’s predictions 
than clinical history variables (for example, prior hospital utilisation).""")

    st.markdown("---")

    # -----------------------------
    # Model diagnostics (PR curve + threshold-F1)
    # -----------------------------
    st.subheader("Model diagnostics")

    diag_col1, diag_col2 = st.columns(2)

    with diag_col1:
        fig_pr = plot_pr_curve(y_true, y_score)
        st.plotly_chart(fig_pr, use_container_width=True)

        st.markdown("""Because readmissions are rare (~2% prevalence), the precision–recall curve is more informative
than ROC-based metrics. The model’s average precision (~0.58) is substantially higher than the
random baseline (expected AP equal to prevalence at ~0.02), indicating that the model ranks
high-risk patients far more effectively than random selection.""")

    with diag_col2:
        fig_f1 = plot_f1_vs_threshold(y_true, y_score)
        st.plotly_chart(fig_f1, use_container_width=True)

        st.markdown("""This plot shows how the F1 score varies with the classification threshold, helping to select 
an operating point that balances precision and recall. The model achieves its highest F1 score (~0.55) 
at a threshold of ~0.28, indicating a balanced compromise between sensitivity and precision while 
limiting false positives.""")

    st.markdown("---")

    # -----------------------------
    # Feature importance
    # -----------------------------
    st.subheader("Feature importance")

    # Try to access the underlying XGB model if the pipeline wraps it
    xgb_model = pipeline
    fig_fi = plot_feature_importance(xgb_model, feature_cols)

    if fig_fi is not None:
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Feature importances not available on this model object.")

    st.markdown("""The number of prior hospital spells is the most influential feature in the model, 
followed by the proportion of emergency admissions and mean length of stay. 
Demographic variables such as age, sex, ethnicity, and IMD have substantially lower importance, 
indicating that recent utilisation patterns drive the model’s predictions more strongly than 
demographic characteristics.""")

    st.markdown("---")

    # -----------------------------
    # High-risk cohort explorer
    # -----------------------------
    st.subheader("High-risk cohort explorer")
    st.caption(
        "Filter and inspect patients above a chosen risk threshold (for audit and exploration).")

    st.markdown("""This table lists patients whose predicted risk exceeds the chosen threshold.
It enables case-level review and supports operational planning (e.g., targeted follow-ups).
Sorting by risk score highlights the most clinically complex patients,
often with high spell counts and mixed respiratory histories.""")

    with st.container(border=True):
        # Controls (kept above the table to avoid horizontal scrolling)
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            thr = st.slider(
                "High-risk threshold",
                0.0, 1.0, float(best_thr), 0.01,
                help="Patients with risk_score ≥ threshold are included below."
            )
        with c2:
            top_n = st.number_input(
                "Max rows to display",
                min_value=25, max_value=500, value=200, step=25,
                help="Keeps the table responsive."
            )

        st.caption(f"Default starts at best F1 threshold ({best_thr:.2f}).")

        df_high = (
            df[df["risk_score"] >= thr]
            .copy()
            .sort_values("risk_score", ascending=False)
        )

        st.metric(
            "High-risk patients",
            f"{len(df_high):,}",
            help="Count of patients meeting the current threshold.",
        )

        cols = [
            "risk_score",
            "readmit_30d",
            "n_spells",
            "pct_emerg",
            "respiratory_group_mode",
            "imd_quintile",
            "age",
            "sex",
            "ethnicity",
        ]
        cols = [c for c in cols if c in df_high.columns]

        df_disp = df_high[cols].head(int(top_n)).rename(columns={
            "risk_score": "Risk score",
            "readmit_30d": "Readmitted 30 days",
            "n_spells": "Prior spells",
            "pct_emerg": "Emergency proportion",
            "respiratory_group_mode": "Respiratory group",
            "imd_quintile": "IMD quintile",
            "sex": "Sex",
        })

        # Round ages to whole integers for display
        if "age" in df_disp.columns:
            df_disp["age"] = pd.to_numeric(
                df_disp["age"], errors="coerce").round().astype("Int64")

        st.dataframe(
            df_disp,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Risk score": st.column_config.ProgressColumn(
                    "Risk score", min_value=0.0, max_value=1.0, format="%.3f"
                ),
                "Readmitted 30 days": st.column_config.CheckboxColumn("Readmitted 30 days"),
                "Emergency proportion": st.column_config.NumberColumn("Emergency proportion", format="%.1f%%"),
            },
        )

        st.download_button(
            "Download filtered cohort (CSV)",
            data=df_high.to_csv(index=False).encode("utf-8"),
            file_name=f"high_risk_cohort_thr_{thr:.2f}.csv",
            mime="text/csv",
            help="Exports the currently filtered cohort.",
        )


if __name__ == "__main__":
    main()
