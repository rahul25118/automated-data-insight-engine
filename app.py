import streamlit as st
import pandas as pd
import numpy as np
import io

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
    page_title="Data Analyst Workbench | Enterprise EDA",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Enterprise Data Analyst Workbench")
st.markdown("प्रोफेशनल डेटा एनालिस्ट्स के लिए हाई-स्पीड स्टैटिस्टिकल प्रोफाइलिंग, आउटलायर हैंडलिंग, ट्रांसफॉर्मेशन और विज़ुअलाइज़ेशन इंजन।")

# Sidebar
st.sidebar.header("📁 डेटा इनपुट")
uploaded_file = st.sidebar.file_uploader("CSV या Excel फ़ाइल अपलोड करें", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    if "data" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
        raw_df, err = load_data(uploaded_file)
        if err:
            st.error(f"फ़ाइल लोड करने में त्रुटि: {err}")
            st.stop()
        st.session_state["data"] = raw_df
        st.session_state["file_name"] = uploaded_file.name
        st.sidebar.success("फ़ाइल लोड हो गई!")

    df = st.session_state["data"]
    metrics = get_basic_metrics(df)

    # Top KPI Metrics
    st.subheader("📌 डेटा हेल्थ समरी (Overview)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("कुल पंक्तियाँ (Rows)", f"{metrics['rows']:,}")
    c2.metric("कुल कॉलम (Columns)", metrics['columns'])
    c3.metric("डुप्लिकेट पंक्तियाँ", metrics['duplicate_rows'])
    c4.metric("मिसिंग सेल्स (%)", f"{metrics['missing_percent']}%")

    # Workbench Tabs (Pure Analyst Focused)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 डेटा प्रिव्यू & स्कीमा",
        "📐 स्टैटिस्टिकल डायग्नोस्टिक्स",
        "🛠️ क्लीनिंग & आउटलायर ट्रीटमेंट",
        "⚡ फ़ीचर ट्रांसफॉर्मेशन",
        "📊 विज़ुअलाइज़ेशन लैब",
        "💾 एक्सपोर्ट & एग्जीक्यूटिव रिपोर्ट"
    ])

    # Tab 1: Preview & Schema
    with tab1:
        st.markdown("### फर्स्ट 5 रोज़")
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown("### डिस्क्रिप्टिव स्टैटिस्टिक्स (Quantiles & Moments)")
        st.dataframe(df.describe(include="all").T, use_container_width=True)

    # Tab 2: Statistical Diagnostics (Analyst Specialty)
    with tab2:
        st.markdown("### 📐 स्टैटिस्टिकल शेप & नॉर्मेलिटी डायग्नोस्टिक्स")
        st.caption("Skewness, Kurtosis और डिस्ट्रीब्यूशन शेप देखकर तय करें कि फीचर को किस ट्रांसफॉर्मेशन (Log/Box-Cox) की जरूरत है।")
        stats_df = get_statistical_diagnostics(df)
        if not stats_df.empty:
            st.dataframe(stats_df, use_container_width=True)
        else:
            st.info("डेटासेट में कोई न्यूमेरिकल कॉलम नहीं है।")

        st.markdown("---")
        st.markdown("### 🔍 मिसिंग वैल्यू ऑडिट")
        missing_df = get_missing_summary(df)
        if not missing_df.empty:
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("✓ कोई मिसिंग वैल्यूज नहीं पाई गईं!")

    # Tab 3: Cleaning & Outliers
    with tab3:
        st.markdown("### 🛠️ क्लीनिंग और आउटलायर ट्रीटमेंट")
        c_left, c_right = st.columns(2)

        with c_left:
            st.markdown("#### 1. Smart 1-Click Auto-Clean")
            st.caption("Skewness के आधार पर Mean/Median, Mode इम्प्यूटेशन, स्ट्रिंग ट्रिमिंग और डुप्लिकेट्स को हटाता है।")
            if st.button("🚀 Run Auto-Clean", type="primary"):
                cleaned_data, clean_logs = smart_auto_clean(df)
                st.session_state["data"] = cleaned_data
                st.success("डेटा ऑटो-क्लीन हो गया!")
                with st.expander("📝 क्लीनिंग लॉग्स", expanded=True):
                    for l in clean_logs:
                        st.write(l)
                st.rerun()

            st.markdown("---")
            st.markdown("#### 2. मैन्युअल इम्प्यूटेशन")
            missing_cols = df.columns[df.isnull().any()].tolist()
            if missing_cols:
                col_to_fix = st.selectbox("मिसिंग कॉलम चुनें:", missing_cols)
                is_num = pd.api.types.is_numeric_dtype(df[col_to_fix])
                strats = ["Mean", "Median", "Mode", "Constant Value", "Drop Rows"] if is_num else ["Mode", "Constant Value", "Drop Rows"]
                strat = st.selectbox("इम्प्यूटेशन मेथड:", strats)
                c_val = st.text_input("कस्टम वैल्यू:") if strat == "Constant Value" else None
                if st.button("Apply Imputation"):
                    st.session_state["data"] = impute_missing_values(df, col_to_fix, strat, c_val)
                    st.success(f"'{col_to_fix}' पर {strat} लागू हो गया!")
                    st.rerun()
            else:
                st.info("कोई मिसिंग वैल्यू नहीं बची है।")

        with c_right:
            st.markdown("#### 3. IQR Outlier Detection & Capping")
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                out_col = st.selectbox("आउटलायर एनालिसिस के लिए कॉलम:", num_cols)
                out_info = detect_outliers_iqr(df, out_col)
                oc1, oc2 = st.columns(2)
                oc1.metric("आउटलायर काउंट", f"{out_info['count']} ({out_info['percent']}%)")
                oc2.metric("IQR बाउंड्स", f"[{out_info['lower_bound']} , {out_info['upper_bound']}]")
                if out_info["count"] > 0:
                    action = st.radio("ट्रीटमेंट चुनें:", ["Cap (Winsorize)", "Remove Rows"], horizontal=True)
                    if st.button(f"Apply {action} on '{out_col}'"):
                        st.session_state["data"] = handle_outliers(df, out_col, method=action)
                        st.success(f"'{out_col}' पर आउटलायर ट्रीटमेंट लागू हो गया!")
                        st.rerun()
                else:
                    st.success("इस कॉलम में कोई आउटलायर नहीं है।")

    # Tab 4: Feature Transformation (Unique for Analysts)
    with tab4:
        st.markdown("### ⚡ फ़ीचर ट्रांसफॉर्मेशन & स्केलिंग लैब")
        st.caption("रिग्रेशन और मशीन लर्निंग मॉडल्स के लिए फीचर्स को तैयार करें।")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            t_col = st.selectbox("ट्रांसफॉर्म करने के लिए कॉलम चुनें:", num_cols, key="trans_col")
            t_type = st.selectbox("ट्रांसफॉर्मेशन प्रकार:", [
                "Log (ln(x+1))", 
                "Square Root", 
                "Z-Score Standardization", 
                "Min-Max Scaling (0 to 1)"
            ])
            if st.button(f"Apply {t_type}"):
                res_df, err = apply_transformation(df, t_col, t_type)
                if err:
                    st.error(err)
                else:
                    st.session_state["data"] = res_df
                    st.success(f"नया ट्रांसफॉर्म्ड कॉलम सफलतापूर्वक डेटा में जुड़ गया!")
                    st.rerun()
        else:
            st.info("ट्रांसफॉर्मेशन के लिए कोई न्यूमेरिकल कॉलम नहीं है।")

    # Tab 5: Advanced Visualization Lab
    with tab5:
        st.markdown("### 📊 विज़ुअलाइज़ेशन लैब")
        v_tab1, v_tab2, v_tab3, v_tab4 = st.tabs([
            "📈 डिस्ट्रीब्यूशन & प्रोबेबिलिटी",
            "🔍 Bivariate Scatter & Trendline",
            "🔥 कोरिलेशन मैट्रिक्स",
            "📊 कैटेगोरिकल फ्रीक्वेंसी"
        ])

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        with v_tab1:
            if num_cols:
                d_col = st.selectbox("फीचर चुनें:", num_cols, key="dist_c")
                d_type = st.radio("प्लॉट प्रकार:", ["Histogram", "Violin Plot", "Box Plot"], horizontal=True)
                st.plotly_chart(plot_distribution(df, d_col, d_type), use_container_width=True)
                st.plotly_chart(plot_ecdf(df, d_col), use_container_width=True)

        with v_tab2:
            if len(num_cols) >= 2:
                sc1, sc2, sc3 = st.columns(3)
                x_c = sc1.selectbox("X-Axis (Feature):", num_cols, index=0)
                y_c = sc2.selectbox("Y-Axis (Target/Feature):", num_cols, index=1 if len(num_cols) > 1 else 0)
                color_opts = ["None"] + cat_cols + num_cols
                hue_c = sc3.selectbox("Color / Dimension:", color_opts)
                trend_chk = st.checkbox("OLS Regression Trendline जोड़ें", value=True)
                st.plotly_chart(plot_bivariate_scatter(df, x_c, y_c, hue_c, trend_chk), use_container_width=True)
            else:
                st.info("स्कैटर प्लॉट के लिए कम से कम 2 न्यूमेरिकल कॉलम चाहिए।")

        with v_tab3:
            if len(num_cols) >= 2:
                corr_m = st.radio("कोरिलेशन मेथड:", ["pearson", "spearman"], horizontal=True)
                st.plotly_chart(plot_correlation_heatmap(df, corr_m), use_container_width=True)
            else:
                st.info("कोरिलेशन के लिए कम से कम 2 न्यूमेरिकल कॉलम चाहिए।")

        with v_tab4:
            if cat_cols:
                cf_col = st.selectbox("कैटेगोरिकल फीचर चुनें:", cat_cols)
                top_k = st.slider("शीर्ष N श्रेणियां दिखाएं:", 5, 30, 15)
                st.plotly_chart(plot_categorical_frequency(df, cf_col, top_k), use_container_width=True)
            else:
                st.info("डेटा में कोई कैटेगोरिकल कॉलम नहीं है।")

    # Tab 6: Export & HTML Report
    with tab6:
        st.markdown("### 💾 डेटा & स्टैटिस्टिकल रिपोर्ट एक्सपोर्ट")
        e_col1, e_col2 = st.columns(2)
        with e_col1:
            st.markdown("#### 1. प्रोसेस किया गया डेटासेट डाउनलोड")
            drop_dups = st.checkbox("डुप्लिकेट पंक्तियाँ हटाकर एक्सपोर्ट करें", value=False)
            exp_df = df.drop_duplicates() if drop_dups else df
            st.download_button(
                "📥 डाउनलोड प्रोसेस्ड CSV",
                exp_df.to_csv(index=False).encode("utf-8"),
                "processed_analyst_data.csv",
                "text/csv"
            )

        with e_col2:
            st.markdown("#### 2. एग्जीक्यूटिव स्टैटिस्टिकल रिपोर्ट")
            stats_table = get_statistical_diagnostics(df)
            stats_html = stats_table.to_html(classes="table", index=False) if not stats_table.empty else ""
            miss_table = get_missing_summary(df)
            miss_html = miss_table.to_html(classes="table", index=False) if not miss_table.empty else ""
            rep_html = generate_html_report(metrics, stats_html, miss_html)
            st.download_button(
                "📄 डाउनलोड HTML/PDF रिपोर्ट",
                rep_html.encode("utf-8"),
                "Analyst_Diagnostic_Report.html",
                "text/html"
            )
else:
    st.info("शुरू करने के लिए साइडबार से CSV या Excel फ़ाइल अपलोड करें।")
