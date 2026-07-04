"""RetailPulse dashboard: customer segmentation + churn risk."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

_root = Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))

import mlflow  # noqa: E402
import mlflow.xgboost  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import streamlit as st  # noqa: E402

from src.dashboard.data_loader import (  # noqa: E402
    ROOT_DIR,
    load_customer_risk_scores,
    load_customer_segments,
    load_shap_importance,
)

LOCAL_CHURN_MODEL_DIR = (
    ROOT_DIR / "data" / "features" / "churn_artifacts" / "models" / "tuned_churn_model"
)

st.set_page_config(page_title="Customers & Churn — RetailPulse", page_icon="👥", layout="wide")
st.title("👥 Customer Segmentation & Churn Risk")

segments = load_customer_segments()
risk_scores = load_customer_risk_scores()

if segments is None:
    st.info("Run `python -m src.models.run_segmentation` to generate customer segments.")
    st.stop()

st.subheader("Segments")
seg_counts = segments["segment_name"].value_counts()
col1, col2 = st.columns([1, 2])
with col1:
    st.bar_chart(seg_counts, color="#2A6F6F", horizontal=True)
with col2:
    profile = (
        segments.groupby("segment_name")[["recency_days", "frequency", "monetary"]].mean().round(1)
    )
    profile["customers"] = seg_counts
    st.dataframe(profile.sort_values("monetary", ascending=False))

st.divider()

if risk_scores is None:
    st.info(
        "Run `python -m src.models.score_customers` to score the full customer base"
    )
    st.stop()

st.subheader("Churn risk by segment")
merged = segments.merge(
    risk_scores[["customer_id", "churn_probability"]], on="customer_id", how="inner"
)
st.caption(
    f"{len(merged):,} of {len(segments):,} segmented customers were also active as of the churn "
    "model's cutoff and have a risk score; the rest weren't yet customers at that point."
)
risk_by_segment = (
    merged.groupby("segment_name")["churn_probability"].mean().sort_values(ascending=False)
)
st.bar_chart(risk_by_segment, color="#C44E52", horizontal=True)

st.divider()
st.subheader("Highest-risk customers")

col1, col2 = st.columns(2)
with col1:
    segment_filter = st.multiselect(
        "Filter by segment", options=sorted(merged["segment_name"].unique()), default=[]
    )
with col2:
    risk_threshold = st.slider("Minimum churn probability", 0.0, 1.0, 0.5, 0.05)

filtered = merged[merged["churn_probability"] >= risk_threshold]
if segment_filter:
    filtered = filtered[filtered["segment_name"].isin(segment_filter)]
filtered = filtered.sort_values("churn_probability", ascending=False)

display_cols = [
    "customer_id",
    "segment_name",
    "churn_probability",
    "recency_days",
    "frequency",
    "monetary",
]
st.dataframe(
    filtered[display_cols]
    .head(200)
    .style.format({"churn_probability": "{:.1%}", "monetary": "£{:.0f}"}),
    width="stretch",
)
st.caption(
    f"Showing top {min(200, len(filtered))} of {len(filtered):,} customers "
)

st.download_button(
    "Download list (CSV)",
    filtered[display_cols].to_csv(index=False),
    file_name="churn_risk_customers.csv",
    mime="text/csv",
)

st.divider()
st.subheader("Why is this customer at risk?")


@st.cache_resource
def _load_churn_model_and_features():
    if LOCAL_CHURN_MODEL_DIR.exists():
        try:
            model = mlflow.xgboost.load_model(str(LOCAL_CHURN_MODEL_DIR))
            return model, model.get_booster().feature_names
        except Exception:
            # Fall back to MLflow tracking lookup if the local artifact is invalid.
            pass

    mlflow.set_tracking_uri(f"file:{ROOT_DIR / 'mlruns'}")
    experiment = mlflow.get_experiment_by_name("churn_prediction")
    if experiment is None:
        return None, []
    try:
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.mlflow.runName = 'day11_churn_tuning'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs.empty:
            return None, []
        model = mlflow.xgboost.load_model(f"runs:/{runs.iloc[0]['run_id']}/tuned_churn_model")
        return model, model.get_booster().feature_names
    except Exception:
        return None, []


model, feature_names = _load_churn_model_and_features()
shap_importance = load_shap_importance()

if model is None:
    st.info(
        "Run `python -m src.models.run_churn_tuning` to train and save the tuned churn model."
    )
    st.stop()

selected_id = st.selectbox(
    "Customer ID", options=filtered["customer_id"].head(200).tolist(), key="customer_selector"
)
if selected_id is not None:
    customer_row = risk_scores[risk_scores["customer_id"] == selected_id][feature_names]
    explainer = shap.TreeExplainer(model)
    sv = explainer(customer_row)

    contrib = pd.Series(sv.values[0], index=feature_names).sort_values()
    st.bar_chart(contrib, color="#2A6F6F", horizontal=True)


if shap_importance is not None:
    with st.expander("Global feature importance"):
        st.dataframe(shap_importance.set_index("feature"))
