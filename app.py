import streamlit as st
import pandas as pd
from engine.statistics import compute_column_metrics, run_one_sample_ttest
from engine.anomaly import detect_iqr_outliers, detect_modified_zscore
from engine.time_series import analyze_trend_walk_forward
from engine.aggregations import compute_group_aggregations

st.set_page_config(page_title="A — Data Insight Engine", layout="wide")

st.title("A — Automated Deterministic Data Insight Engine")
st.caption("Zero-hallucination, pure-statistical analytical engine for structured tabular data.")

uploaded_file = st.file_uploader("Upload your structured CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("1. Raw Dataset Preview")
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(10), use_container_width=True)

    num_cols = df.select_dtypes(include=['number']).columns.tolist()

    if num_cols:
        st.subheader("2. Deterministic Statistical Profiling")
        selected_col = st.selectbox("Select numeric column to inspect:", num_cols)
        
        col1, col2 = st.columns(2)
        metrics = compute_column_metrics(df[selected_col])
        with col1:
            st.write("**Descriptive Metrics:**")
            st.json(metrics)

        with col2:
            st.write("**Outlier Isolation (IQR & Modified Z):**")
            outliers_iqr = detect_iqr_outliers(df[selected_col])
            outliers_modz = detect_modified_zscore(df[selected_col])
            st.write(f"IQR Outliers Detected: **{outliers_iqr['count']}**")
            st.write(f"Modified Z-Score Outliers: **{outliers_modz['count']}**")
            if outliers_iqr['count'] > 0:
                st.write(f"Bounds: {outliers_iqr['bounds']}")

        st.subheader("3. Walk-Forward Trend & Time Series")
        ts_metrics = analyze_trend_walk_forward(df[selected_col])
        st.json(ts_metrics)

        st.subheader("4. Inferential Testing (One-Sample t-Test)")
        mu0 = st.number_input("Hypothesized Mean (mu0):", value=0.0)
        if st.button("Run Hypothesis Test"):
            ttest_res = run_one_sample_ttest(df[selected_col], mu0=mu0)
            st.json(ttest_res)

    st.subheader("5. Group Aggregations")
    aggs = compute_group_aggregations(df)
    if aggs:
        st.json(aggs)
    else:
        st.info("No categorical + numerical pairs found for group aggregation.")
else:
    st.info("Please upload a CSV file to begin analysis.")
