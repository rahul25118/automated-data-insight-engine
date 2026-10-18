import streamlit as st
import pandas as pd
import numpy as np
from modules.data_engine import load_data, get_basic_metrics, get_missing_summary
from modules.viz_engine import plot_correlation_heatmap, plot_distribution, plot_categorical_frequency

st.set_page_config(
    page_title="Automated EDA & Data Insights",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Automated EDA & Data Profiling App")
st.markdown("डेटा अपलोड करें और इंस्टेंट एक्सप्लोरेटरी डेटा एनालिसिस (EDA) रिपोर्ट प्राप्त करें।")

uploaded_file = st.sidebar.file_uploader("CSV या Excel फ़ाइल अपलोड करें", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    df, err = load_data(uploaded_file)
    if err:
        st.error(f"फ़ाइल लोड करने में त्रुटि: {err}")
    else:
        st.sidebar.success("फ़ाइल सफलतापूर्वक लोड हो गई!")

        # 1. KPI Metrics
        metrics = get_basic_metrics(df)
        st.subheader("📌 मुख्य डेटा मेट्रिक्स (Overview)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("कुल पंक्तियाँ (Rows)", f"{metrics['rows']:,}")
        c2.metric("कुल कॉलम (Columns)", metrics['columns'])
        c3.metric("डुप्लिकेट पंक्तियाँ", metrics['duplicate_rows'])
        c4.metric("मिसिंग सेल्स (%)", f"{metrics['missing_percent']}%")

        # 2. Tabs for Structured Insights
        tab1, tab2, tab3, tab4 = st.tabs(["📋 डेटा पूर्वावलोकन", "🔍 मिसिंग वैल्यूज़", "📈 विज़ुअलाइज़ेशन", "💾 डेटा एक्सपोर्ट"])

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

        with tab4:
            st.markdown("### डेटा एक्सपोर्ट विकल्प")
            drop_dups = st.checkbox("डुप्लिकेट पंक्तियाँ हटाएं", value=False)
            export_df = df.drop_duplicates() if drop_dups else df

            csv_data = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 प्रोसेस किया गया CSV डाउनलोड करें",
                data=csv_data,
                file_name="processed_data.csv",
                mime="text/csv"
            )
else:
    st.info("शुरू करने के लिए साइडबार से CSV या Excel फ़ाइल अपलोड करें।")
