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
    smart_auto_clean
)
from modules.viz_engine import plot_correlation_heatmap, plot_distribution, plot_categorical_frequency
from modules.ai_engine import generate_data_narrative, query_data_with_llm
from modules.report_engine import generate_html_report

st.set_page_config(
    page_title="Automated EDA & DeepSeek AI Insights Engine",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Automated EDA & DeepSeek Insights Engine")
st.markdown("डेटा अपलोड करें, 1-क्लिक स्मार्ट क्लीनिंग चलाएं, आउटलायर्स फिक्स करें और DeepSeek AI से अल्ट्रा-फास्ट इनसाइट्स व कोड क्वेरी पाएं।")

# Sidebar
st.sidebar.header("⚙️ सेटिंग्स और इनपुट्स")
uploaded_file = st.sidebar.file_uploader("CSV या Excel फ़ाइल अपलोड करें", type=["csv", "xlsx", "xls"])

st.sidebar.markdown("---")
st.sidebar.subheader("🔑 DeepSeek API Key (BYOK)")
deepseek_api_key = st.sidebar.text_input(
    "अपनी DeepSeek API Key दर्ज करें:",
    type="password",
    help="की प्राप्त करने के लिए platform.deepseek.com पर जाएं"
)
st.sidebar.caption("👉 [Get DeepSeek API Key](https://platform.deepseek.com/)")

if uploaded_file is not None:
    if "data" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
        raw_df, err = load_data(uploaded_file)
        if err:
            st.error(f"फ़ाइल लोड करने में त्रुटि: {err}")
            st.stop()
        st.session_state["data"] = raw_df
        st.session_state["file_name"] = uploaded_file.name
        st.session_state["ai_summary"] = ""
        st.sidebar.success("फ़ाइल सफलतापूर्वक लोड हो गई!")

    df = st.session_state["data"]

    # KPI Metrics
    metrics = get_basic_metrics(df)
    st.subheader("📌 मुख्य डेटा मेट्रिक्स (Overview)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("कुल पंक्तियाँ (Rows)", f"{metrics['rows']:,}")
    c2.metric("कुल कॉलम (Columns)", metrics['columns'])
    c3.metric("डुप्लिकेट पंक्तियाँ", metrics['duplicate_rows'])
    c4.metric("मिसिंग सेल्स (%)", f"{metrics['missing_percent']}%")

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 डेटा प्रिव्यू", 
        "🔍 मिसिंग वैल्यूज़", 
        "🛠️ स्मार्ट क्लीनिंग & आउटलायर्स", 
        "📈 विज़ुअलाइज़ेशन", 
        "🤖 DeepSeek AI इनसाइट्स & चैट",
        "💾 डेटा & रिपोर्ट एक्सपोर्ट"
    ])

    with tab1:
        st.markdown("### डेटा हेड (First 5 Rows)")
        st.dataframe(df.head(), use_container_width=True)
        st.markdown("### डिस्क्रिप्टिव स्टैटिस्टिक्स")
        st.dataframe(df.describe(include="all").T, use_container_width=True)

    with tab2:
        st.markdown("### मिसिंग वैल्यू समरी")
        missing_df = get_missing_summary(df)
        if not missing_df.empty:
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("डेटासेट में कोई भी मिसिंग वैल्यू नहीं है!")

    with tab3:
        st.markdown("### 🛠️ डेटा क्लीनिंग और आउटलायर मैनेजमेंट")
        
        # Section A: 1-Click Smart Auto Clean
        st.subheader("⚡ 1. Smart 1-Click Auto-Clean")
        st.caption("यह स्वचालित रूप से Skewness के आधार पर Mean/Median इम्प्यूटेशन, कैटेगोरिकल Mode, स्ट्रिंग ट्रिमिंग और डुप्लिकेट्स को रिमूव करता है।")
        if st.button("🚀 Run Smart Auto-Clean", type="primary"):
            cleaned_data, clean_logs = smart_auto_clean(df)
            st.session_state["data"] = cleaned_data
            st.success("डेटासेट को सफलतापूर्वक ऑटो-क्लीन किया गया!")
            with st.expander("📝 क्लीनिंग में किए गए सुधार (Logs)", expanded=True):
                for log_item in clean_logs:
                    st.write(log_item)
            st.rerun()

        st.markdown("---")

        # Section B: Manual Imputation
        st.subheader("🎯 2. मैन्युअल मिसिंग वैल्यू इम्प्यूटेशन")
        missing_cols = df.columns[df.isnull().any()].tolist()
        if not missing_cols:
            st.info("डेटासेट में कोई मिसिंग वैल्यू नहीं बची है।")
        else:
            col_to_fix = st.selectbox("कॉलम चुनें:", missing_cols, key="manual_imp_col")
            is_num = pd.api.types.is_numeric_dtype(df[col_to_fix])
            strategies = ["Mean", "Median", "Mode", "Constant Value", "Drop Rows"] if is_num else ["Mode", "Constant Value", "Drop Rows"]
            strategy = st.selectbox("मेथड चुनें:", strategies, key="manual_imp_strat")
            custom_val = st.text_input("कस्टम वैल्यू दर्ज करें:") if strategy == "Constant Value" else None

            if st.button("Apply Manual Imputation"):
                st.session_state["data"] = impute_missing_values(df, col_to_fix, strategy, custom_val)
                st.success(f"'{col_to_fix}' पर '{strategy}' लागू किया गया!")
                st.rerun()

        st.markdown("---")

        # Section C: Outlier Detection and Capping
        st.subheader("📊 3. IQR Outlier Detection & Handling")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            st.info("डेटा में कोई न्यूमेरिकल कॉलम नहीं है।")
        else:
            outlier_col = st.selectbox("आउटलायर्स चेक करने के लिए कॉलम चुनें:", num_cols, key="outlier_col_select")
            out_info = detect_outliers_iqr(df, outlier_col)
            
            oc1, oc2, oc3, oc4 = st.columns(4)
            oc1.metric("आउटलायर काउंट", out_info["count"])
            oc2.metric("आउटलायर (%)", f"{out_info['percent']}%")
            oc3.metric("लोअर बाउंड (Q1 - 1.5*IQR)", out_info["lower_bound"])
            oc4.metric("अपर बाउंड (Q3 + 1.5*IQR)", out_info["upper_bound"])

            if out_info["count"] > 0:
                out_action = st.radio("आउटलायर्स पर क्या कार्रवाई करें?", ["Cap (Winsorize)", "Remove Rows"], horizontal=True)
                if st.button(f"Apply Outlier Treatment on '{outlier_col}'"):
                    st.session_state["data"] = handle_outliers(df, outlier_col, method=out_action)
                    st.success(f"'{outlier_col}' पर {out_action} सफलतापूर्वक लागू किया गया!")
                    st.rerun()
            else:
                st.success(f"'{outlier_col}' में कोई आउटलायर नहीं पाया गया।")

    with tab4:
        st.markdown("### डेटा विज़ुअलाइज़ेशन")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        if numeric_cols:
            col_sel = st.selectbox("न्यूमेरिकल कॉलम चुनें (Distribution):", numeric_cols)
            plot_t = st.radio("प्लॉट का प्रकार:", ["Histogram", "Box Plot"], horizontal=True)
            fig_dist = plot_distribution(df, col_sel, plot_t)
            st.plotly_chart(fig_dist, use_container_width=True)

            if len(numeric_cols) >= 2:
                st.markdown("#### कोरिलेशन मैट्रिक्स")
                fig_corr = plot_correlation_heatmap(df)
                if fig_corr:
                    st.plotly_chart(fig_corr, use_container_width=True)

        if cat_cols:
            st.markdown("#### कैटेगोरिकल फ़ीचर एनालिसिस")
            cat_sel = st.selectbox("कैटेगोरिकल कॉलम चुनें:", cat_cols)
            fig_cat = plot_categorical_frequency(df, cat_sel)
            st.plotly_chart(fig_cat, use_container_width=True)

    with tab5:
        st.markdown("### 🤖 DeepSeek AI एग्जीक्यूटिव समरी")
        if not deepseek_api_key:
            st.warning("⚠️ कृपया साइडबार में अपनी DeepSeek API Key दर्ज करें।")
        else:
            if st.button("⚡ Generate AI Summary Report", type="primary"):
                with st.spinner("DeepSeek डेटा का विश्लेषण कर रहा है..."):
                    summary_payload = {
                        "rows": metrics["rows"],
                        "columns": metrics["columns"],
                        "missing_percent": metrics["missing_percent"],
                        "duplicate_rows": metrics["duplicate_rows"],
                        "numerical_cols": metrics["numerical_cols"],
                        "categorical_cols": metrics["categorical_cols"],
                        "columns_list": list(df.columns)
                    }
                    report = generate_data_narrative(summary_payload, deepseek_api_key)
                    st.session_state["ai_summary"] = report
                    st.markdown("---")
                    st.markdown(report)
            elif st.session_state.get("ai_summary"):
                st.markdown("---")
                st.markdown(st.session_state["ai_summary"])

            st.markdown("---")
            st.markdown("### 💬 Chat with your Data (DeepSeek Code Engine)")
            st.caption("उदा. 'Top 5 rows with highest value', 'Find average value of column X'")
            
            user_query = st.text_input("अपने डेटा से सवाल पूछें:")
            if st.button("🔍 सवाल पूछें (Ask Data)"):
                if not user_query.strip():
                    st.info("कृपया कोई सवाल टाइप करें।")
                else:
                    with st.spinner("DeepSeek क्वेरी समझ रहा है और Pandas कोड चला रहा है..."):
                        buf = io.StringIO()
                        df.info(buf=buf)
                        schema_str = f"Columns & Types:\n{buf.getvalue()}\n\nSample Data (First 3 rows):\n{df.head(3).to_dict(orient='records')}"

                        res_dict = query_data_with_llm(user_query, schema_str, deepseek_api_key)
                        if "error" in res_dict:
                            st.error(res_dict["error"])
                        else:
                            st.info(f"💡 **विश्लेषण:** {res_dict.get('explanation', '')}")
                            generated_code = res_dict.get("code", "")
                            
                            with st.expander("🛠️ जनरेट किया गया Pandas कोड देखें"):
                                st.code(generated_code, language="python")

                            local_vars = {"df": df, "pd": pd, "np": np}
                            try:
                                exec(generated_code, {}, local_vars)
                                query_result = local_vars.get("result", None)

                                if query_result is not None:
                                    st.markdown("#### 📊 परिणाम:")
                                    if isinstance(query_result, (pd.DataFrame, pd.Series)):
                                        st.dataframe(query_result, use_container_width=True)
                                    else:
                                        st.write(query_result)
                                else:
                                    st.warning("क्वेरी रन हो गई लेकिन कोई 'result' वेरिएबल वापस नहीं आया।")
                            except Exception as exec_err:
                                st.error(f"कोड चलाने में त्रुटि: {exec_err}")

    with tab6:
        st.markdown("### 💾 डेटा & रिपोर्ट एक्सपोर्ट")
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("#### 1. क्लीन्ड डेटा डाउनलोड")
            drop_dups = st.checkbox("डुप्लिकेट पंक्तियाँ हटाएं", value=False)
            export_df = df.drop_duplicates() if drop_dups else df
            csv_data = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 डाउनलोड क्लीन्ड CSV",
                data=csv_data,
                file_name="cleaned_imputed_data.csv",
                mime="text/csv"
            )

        with c_right:
            st.markdown("#### 2. एग्जीक्यूटिव बिज़नेस रिपोर्ट")
            missing_df = get_missing_summary(df)
            missing_html = missing_df.to_html(classes="table", index=False) if not missing_df.empty else ""
            html_report = generate_html_report(
                metrics=metrics,
                ai_summary=st.session_state.get("ai_summary", ""),
                missing_summary_html=missing_html
            )

            st.download_button(
                label="📄 डाउनलोड एग्जीक्यूटिव HTML/PDF रिपोर्ट",
                data=html_report.encode("utf-8"),
                file_name="Executive_Data_Report.html",
                mime="text/html"
            )
else:
    st.info("शुरू करने के लिए साइडबार से CSV या Excel फ़ाइल अपलोड करें।")
