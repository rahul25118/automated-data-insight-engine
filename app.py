import os
import sys

# Ensure current directory and modules are explicitly discoverable in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import streamlit as st
import pandas as pd
import numpy as np

from modules.data_engine import (
    load_data, 
    get_basic_metrics, 
    get_missing_summary, 
    impute_missing_values,
    detect_outliers_iqr,
    handle_outliers,
    smart_auto_clean,
    get_statistical_diagnostics,
    apply_transformation
)
from modules.viz_engine import (
    plot_distribution, 
    plot_bivariate_scatter, 
    plot_correlation_heatmap, 
    plot_categorical_frequency,
    plot_ecdf
)
from modules.report_engine import generate_html_report

st.set_page_config(
    page_title="A",
    page_icon="⚡",
    layout="wide"
)

# Brand: Sirf A
st.markdown("<h1 style='font-size: 4rem; font-weight: 900; margin-bottom: 0;'>A</h1>", unsafe_allow_html=True)
st.caption("A / Professional Data Analyst Workbench")

# Sidebar
st.sidebar.markdown("## **A**")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    if "data" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
        raw_df, err = load_data(uploaded_file)
        if err:
            st.error(f"Error loading file: {err}")
            st.stop()
        st.session_state["data"] = raw_df
        st.session_state["file_name"] = uploaded_file.name
        st.sidebar.success("File loaded successfully")

    df = st.session_state["data"]
    metrics = get_basic_metrics(df)

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{metrics['rows']:,}")
    c2.metric("Columns", metrics['columns'])
    c3.metric("Duplicates", metrics['duplicate_rows'])
    c4.metric("Missing Cells", f"{metrics['missing_percent']}%")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Preview & Schema",
        "📐 Diagnostics",
        "🛠️ Clean & Outliers",
        "⚡ Feature Transforms",
        "📊 Visual Suite",
        "💾 Export & Report"
    ])

    with tab1:
        st.markdown("#### Sample Records")
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown("#### Descriptive Statistics")
        st.dataframe(df.describe(include="all").T, use_container_width=True)

    with tab2:
        st.markdown("#### Statistical Moments & Shape Profiling")
        stats_df = get_statistical_diagnostics(df)
        if not stats_df.empty:
            st.dataframe(stats_df, use_container_width=True)
        else:
            st.info("No numerical features available.")

        st.markdown("---")
        st.markdown("#### Missing Value Audit")
        missing_df = get_missing_summary(df)
        if not missing_df.empty:
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("✓ No missing values detected.")

    with tab3:
        st.markdown("#### Data Remediation & Outlier Treatment")
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("##### 1. Smart Clean")
            if st.button("🚀 Run Smart Clean", type="primary"):
                cleaned_data, clean_logs = smart_auto_clean(df)
                st.session_state["data"] = cleaned_data
                st.success("Dataset cleaned.")
                with st.expander("Logs", expanded=True):
                    for l in clean_logs:
                        st.write(l)
                st.rerun()

            st.markdown("---")
            st.markdown("##### 2. Targeted Imputation")
            missing_cols = df.columns[df.isnull().any()].tolist()
            if missing_cols:
                col_to_fix = st.selectbox("Select column:", missing_cols)
                is_num = pd.api.types.is_numeric_dtype(df[col_to_fix])
                strats = ["Mean", "Median", "Mode", "Constant Value", "Drop Rows"] if is_num else ["Mode", "Constant Value", "Drop Rows"]
                strat = st.selectbox("Imputation method:", strats)
                c_val = st.text_input("Constant value:") if strat == "Constant Value" else None
                if st.button("Apply Imputation"):
                    st.session_state["data"] = impute_missing_values(df, col_to_fix, strat, c_val)
                    st.success(f"Applied {strat} on '{col_to_fix}'.")
                    st.rerun()
            else:
                st.info("No remaining missing values.")

        with col_r:
            st.markdown("##### 3. IQR Outlier Inspection & Capping")
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                out_col = st.selectbox("Feature for outliers:", num_cols)
                out_info = detect_outliers_iqr(df, out_col)
                oc1, oc2 = st.columns(2)
                oc1.metric("Outlier Count", f"{out_info['count']} ({out_info['percent']}%)")
                oc2.metric("IQR Bounds", f"[{out_info['lower_bound']}, {out_info['upper_bound']}]")
                if out_info["count"] > 0:
                    action = st.radio("Outlier Action:", ["Cap (Winsorize)", "Remove Rows"], horizontal=True)
                    if st.button(f"Apply {action} on '{out_col}'"):
                        st.session_state["data"] = handle_outliers(df, out_col, method=action)
                        st.success(f"Outlier action {action} applied.")
                        st.rerun()
                else:
                    st.success(f"No outliers detected in '{out_col}'.")

    with tab4:
        st.markdown("#### Mathematical Transforms & Scalers")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            t_col = st.selectbox("Column to transform:", num_cols, key="trans_col")
            t_type = st.selectbox("Transformation Type:", [
                "Log (ln(x+1))", 
                "Square Root", 
                "Z-Score Standardization", 
                "Min-Max Scaling (0 to 1)"
            ])
            if st.button("Generate Transformed Feature"):
                res_df, err = apply_transformation(df, t_col, t_type)
                if err:
                    st.error(err)
                else:
                    st.session_state["data"] = res_df
                    st.success("Transformed feature added.")
                    st.rerun()
        else:
            st.info("No numerical columns available.")

    with tab5:
        st.markdown("#### Exploratory Visual Suite")
        v_sub1, v_sub2, v_sub3, v_sub4 = st.tabs([
            "📈 Distributions & CDF",
            "🔍 Scatter & Trendline",
            "🔥 Correlation Heatmap",
            "📊 Categorical Frequencies"
        ])

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        with v_sub1:
            if num_cols:
                d_col = st.selectbox("Feature:", num_cols, key="vis_d_col")
                d_type = st.radio("Plot Type:", ["Histogram", "Violin Plot", "Box Plot"], horizontal=True)
                st.plotly_chart(plot_distribution(df, d_col, d_type), use_container_width=True)
                st.plotly_chart(plot_ecdf(df, d_col), use_container_width=True)

        with v_sub2:
            if len(num_cols) >= 2:
                sc1, sc2, sc3 = st.columns(3)
                x_c = sc1.selectbox("X-Axis:", num_cols, index=0)
                y_c = sc2.selectbox("Y-Axis:", num_cols, index=1 if len(num_cols) > 1 else 0)
                color_opts = ["None"] + cat_cols + num_cols
                hue_c = sc3.selectbox("Hue Dimension:", color_opts)
                trend_chk = st.checkbox("Fit OLS Trendline", value=True)
                st.plotly_chart(plot_bivariate_scatter(df, x_c, y_c, hue_c, trend_chk), use_container_width=True)
            else:
                st.info("Requires at least 2 numerical features.")

        with v_sub3:
            if len(num_cols) >= 2:
                corr_m = st.radio("Correlation Method:", ["pearson", "spearman"], horizontal=True)
                st.plotly_chart(plot_correlation_heatmap(df, corr_m), use_container_width=True)
            else:
                st.info("Requires at least 2 numerical features.")

        with v_sub4:
            if cat_cols:
                cf_col = st.selectbox("Categorical Feature:", cat_cols)
                top_k = st.slider("Max categories:", 5, 30, 15)
                st.plotly_chart(plot_categorical_frequency(df, cf_col, top_k), use_container_width=True)
            else:
                st.info("No categorical features detected.")

    with tab6:
        st.markdown("#### Export & Diagnostic Documentation")
        e_col1, e_col2 = st.columns(2)
        with e_col1:
            st.markdown("##### 1. Cleaned Dataset")
            drop_dups = st.checkbox("Deduplicate prior to export", value=False)
            exp_df = df.drop_duplicates() if drop_dups else df
            st.download_button(
                "📥 Download Cleaned CSV",
                exp_df.to_csv(index=False).encode("utf-8"),
                "A_processed_data.csv",
                "text/csv"
            )

        with e_col2:
            st.markdown("##### 2. Diagnostic HTML Report")
            stats_table = get_statistical_diagnostics(df)
            stats_html = stats_table.to_html(classes="table", index=False) if not stats_table.empty else ""
            miss_table = get_missing_summary(df)
            miss_html = miss_table.to_html(classes="table", index=False) if not miss_table.empty else ""
            rep_html = generate_html_report(metrics, stats_html, miss_html)
            st.download_button(
                "📄 Download A Diagnostic Report",
                rep_html.encode("utf-8"),
                "A_Diagnostic_Report.html",
                "text/html"
            )
else:
    st.info("Upload a CSV or Excel file to initialize workbench 'A'.")
