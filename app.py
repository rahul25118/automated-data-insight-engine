import streamlit as st
import pandas as pd
import numpy as np
from modules.data_engine import load_data, get_basic_metrics, get_missing_summary, impute_missing_values
from modules.viz_engine import plot_correlation_heatmap, plot_distribution, plot_categorical_frequency
from modules.ai_engine import generate_data_narrative

st.set_page_config(
    page_title="Automated EDA & AI Insights Engine",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Automated EDA & AI Insights Engine")
st.markdown("डेटा अपलोड करें, मिसिंग वैल्यूज़ फिक्स करें और Gemini AI से इंस्टेंट एग्जीक्यूटिव बिजनेस इनसाइट्स प्राप्त करें।")

# 1. Sidebar Configurations
st.sidebar.header("⚙️ सेटिंग्स और इनपुट्स")
uploaded_file = st.sidebar.file_uploader("CSV या Excel फ़ाइल अपलोड करें", type=["csv", "xlsx", "xls"])

st.sidebar.markdown("---")
st.sidebar.subheader("🔑 Gemini API Key (BYOK)")
gemini_api_key = st.sidebar.text_input(
    "अपनी Gemini API Key दर्ज करें:",
    type="password",
    help="फ्री की पाने के लिए aistudio.google.com पर जाएं"
)
st.sidebar.caption("👉 [Get Free Gemini API Key](https://aistudio.google.com/)")

if uploaded_file is not None:
    if "data" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
        raw_df, err = load_data(uploaded_file)
        if err:
            st.error(f"फ़ाइल लोड करने में त्रुटि: {err}")
            st.stop()
        st.session_state["data"] = raw_df
        st.session_state["file_name"] = uploaded_file.name
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
        "🛠️ मिसिंग वैल्यू इम्प्यूटेशन", 
        "📈 विज़ुअलाइज़ेशन", 
        "🤖 AI बिजनेस इनसाइट्स",
        "💾 डेटा एक्सपोर्ट"
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
        st.markdown("### 🛠️ मिसिंग वैल्यूज़ को हैंडल करें")
        missing_cols = df.columns[df.isnull().any()].tolist()

        if not missing_cols:
            st.success("सारे कॉलम्स क्लीन हैं! कोई मिसिंग वैल्यू नहीं बची।")
        else:
            col_to_fix = st.selectbox("कॉलम चुनें जिसमें मिसिंग वैल्यूज हैं:", missing_cols)
            is_num = pd.api.types.is_numeric_dtype(df[col_to_fix])

            if is_num:
                strategies = ["Mean", "Median", "Mode", "Constant Value", "Drop Rows"]
            else:
                strategies = ["Mode", "Constant Value", "Drop Rows"]

            strategy = st.selectbox("इम्प्यूटेशन मेथड चुनें:", strategies)
            custom_val = None
            if strategy == "Constant Value":
                custom_val = st.text_input("कस्टम वैल्यू दर्ज करें:")

            if st.button("Apply Imputation (लागू करें)"):
                st.session_state["data"] = impute_missing_values(df, col_to_fix, strategy, custom_val)
                st.success(f"'{col_to_fix}' पर '{strategy}' सफलतापूर्वक लागू किया गया!")
                st.rerun()

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
        st.markdown("### 🤖 AI एग्जीक्यूटिव समरी और इनसाइट्स")
        st.info("यह मॉड्यूल आपके पूरे डेटासेट की समरी का विश्लेषण करके बिज़नेस सिफ़ारिशें और रिस्क रिपोर्ट जनरेट करता है।")

        if not gemini_api_key:
            st.warning("⚠️ कृपया साइडबार में अपनी Gemini API Key दर्ज करें।")
        else:
            if st.button("✨ Generate AI Insights Report", type="primary"):
                with st.spinner("AI डेटा का विश्लेषण कर रहा है... कृपया प्रतीक्षा करें..."):
                    summary_payload = {
                        "rows": metrics["rows"],
                        "columns": metrics["columns"],
                        "missing_percent": metrics["missing_percent"],
                        "duplicate_rows": metrics["duplicate_rows"],
                        "numerical_cols": metrics["numerical_cols"],
                        "categorical_cols": metrics["categorical_cols"],
                        "columns_list": list(df.columns)
                    }
                    report = generate_data_narrative(summary_payload, gemini_api_key)
                    st.markdown("---")
                    st.markdown(report)

    with tab6:
        st.markdown("### डेटा एक्सपोर्ट विकल्प")
        drop_dups = st.checkbox("डुप्लिकेट पंक्तियाँ हटाएं", value=False)
        export_df = df.drop_duplicates() if drop_dups else df
        
        csv_data = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 प्रोसेस्ड / क्लीन्ड CSV डाउनलोड करें",
            data=csv_data,
            file_name="cleaned_imputed_data.csv",
            mime="text/csv"
        )
else:
    st.info("शुरू करने के लिए साइडबार से CSV या Excel फ़ाइल अपलोड करें।")
