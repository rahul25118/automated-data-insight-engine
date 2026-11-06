import streamlit as st
import pandas as pd
import io
from engine.statistics import compute_column_metrics, run_one_sample_ttest
from engine.anomaly import detect_iqr_outliers, detect_modified_zscore
from engine.time_series import analyze_trend_walk_forward
from engine.aggregations import compute_group_aggregations

st.set_page_config(page_title="A — Data Insight Engine", layout="wide")

st.title("A — Automated Deterministic Data Insight Engine")
st.caption("Zero-hallucination, pure-statistical analytical engine for structured tabular data.")

# Cache using raw bytes instead of file object to prevent buffer pointer stalls
@st.cache_data(show_spinner="Parsing dataset...")
def parse_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))

uploaded_file = st.file_uploader(
    "Upload your structured CSV dataset", 
    type=["csv"],
    help="Select or drag and drop a valid CSV file"
)

if uploaded_file is not None:
    try:
        # Reset file pointer and extract raw bytes
        uploaded_file.seek(0)
        file_bytes = uploaded_file.getvalue()
        
        if len(file_bytes) == 0:
            st.error("Uploaded CSV file is empty. Please upload a file with data.")
            st.stop()
            
        df = parse_csv_bytes(file_bytes)
        
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        st.stop()

    st.success(f"✓ File loaded successfully: {uploaded_file.name} ({len(file_bytes) / 1024:.1f} KB)")
    
    st.subheader("1. Raw Dataset Preview")
    st.write(f"**Dimensions:** {df.shape[0]:,} rows × {df.shape[1]} columns")
    st.dataframe(df.head(25), height=320, use_container_width=True)

    num_cols = df.select_dtypes(include=['number']).columns.tolist()

    if num_cols:
        st.subheader("2. Deterministic Statistical Profiling")
        selected_col = st.selectbox("Select numeric column to inspect:", num_cols)

        col1, col2 = st.columns(2)
        metrics = compute_column_metrics(df[selected_col])
        outliers_iqr = detect_iqr_outliers(df[selected_col])
        outliers_modz = detect_modified_zscore(df[selected_col])

        with col1:
            st.write("**Descriptive Metrics:**")
            st.json(metrics)

        with col2:
            st.write("**Outlier Isolation (IQR & Modified Z):**")
            st.write(f"IQR Outliers Detected: **{outliers_iqr['count']}**")
            st.write(f"Modified Z-Score Outliers: **{outliers_modz['count']}**")
            if outliers_iqr['count'] > 0:
                st.write(f"Bounds: {outliers_iqr['bounds']}")
                st.write(f"Sample Outliers: {outliers_iqr['values'][:8]}")

        st.subheader("3. Walk-Forward Trend & Time Series")
        ts_metrics = analyze_trend_walk_forward(df[selected_col])
        st.json(ts_metrics)

        st.subheader("4. Inferential Testing (One-Sample t-Test)")
        mu0 = st.number_input("Hypothesized Mean (mu0):", value=0.0, step=0.1)
        if st.button("Run Hypothesis Test"):
            ttest_res = run_one_sample_ttest(df[selected_col], mu0=mu0)
            st.json(ttest_res)
    else:
        st.warning("No numeric columns detected in this CSV for statistical computation.")

    st.subheader("5. Group Aggregations")
    aggs = compute_group_aggregations(df)
    if aggs:
        st.json(aggs)
    else:
        st.info("No categorical + numerical pairs available for group aggregation.")
else:
    st.info("Upload a CSV file above to begin analysis.")
