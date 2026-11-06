import streamlit as st
import pandas as pd
from engine.statistics import compute_column_metrics, run_one_sample_ttest
from engine.anomaly import detect_iqr_outliers, detect_modified_zscore
from engine.time_series import analyze_trend_walk_forward
from engine.aggregations import compute_group_aggregations

st.set_page_config(page_title="A — Data Insight Engine", layout="wide")

st.title("A — Automated Deterministic Data Insight Engine")
st.caption("Zero-hallucination, pure-statistical analytical engine for structured tabular data.")

# Cache CSV parsing to prevent re-reading on every browser interaction
@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

# Cache heavy calculations per column
@st.cache_data(show_spinner=False)
def get_cached_metrics(series: pd.Series):
    return compute_column_metrics(series)

@st.cache_data(show_spinner=False)
def get_cached_anomalies(series: pd.Series):
    return detect_iqr_outliers(series), detect_modified_zscore(series)

@st.cache_data(show_spinner=False)
def get_cached_trend(series: pd.Series):
    return analyze_trend_walk_forward(series)

uploaded_file = st.file_uploader("Upload your structured CSV dataset", type=["csv"])

if uploaded_file is not None:
    with st.spinner("Parsing dataset into memory..."):
        df = load_csv(uploaded_file)

    st.subheader("1. Raw Dataset Preview")
    st.write(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Bounded rendering prevents DOM/canvas freezing on large datasets
    st.dataframe(df.head(25), height=350)
    if len(df) > 25:
        st.caption(f"Showing first 25 rows (out of {len(df):,}) to keep scrolling smooth.")

    num_cols = df.select_dtypes(include=['number']).columns.tolist()

    if num_cols:
        st.subheader("2. Deterministic Statistical Profiling")
        selected_col = st.selectbox("Select numeric column to inspect:", num_cols)

        col1, col2 = st.columns(2)
        metrics = get_cached_metrics(df[selected_col])
        outliers_iqr, outliers_modz = get_cached_anomalies(df[selected_col])

        with col1:
            st.write("**Descriptive Metrics:**")
            st.json(metrics)

        with col2:
            st.write("**Outlier Isolation (IQR & Modified Z):**")
            st.write(f"IQR Outliers Detected: **{outliers_iqr['count']}**")
            st.write(f"Modified Z-Score Outliers: **{outliers_modz['count']}**")
            if outliers_iqr['count'] > 0:
                st.write(f"Bounds: {outliers_iqr['bounds']}")
                st.write(f"Sample Outlier Values: {outliers_iqr['values'][:10]}")

        st.subheader("3. Walk-Forward Trend & Time Series")
        ts_metrics = get_cached_trend(df[selected_col])
        st.json(ts_metrics)

        st.subheader("4. Inferential Testing (One-Sample t-Test)")
        mu0 = st.number_input("Hypothesized Mean (mu0):", value=0.0, step=0.1)
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
