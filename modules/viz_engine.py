import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_correlation_heatmap(df: pd.DataFrame):
    """न्यूमेरिकल कॉलम्स का कोरिलेशन हीटमैप बनाता है"""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return None
    corr = numeric_df.corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Viridis",
        title="फ़ीचर कोरिलेशन हीटमैप (Correlation Heatmap)"
    )
    fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
    return fig

def plot_distribution(df: pd.DataFrame, column: str, plot_type: str = "Histogram"):
    """न्यूमेरिकल कॉलम का डिस्ट्रीब्यूशन प्लॉट बनाता है"""
    if plot_type == "Histogram":
        fig = px.histogram(
            df,
            x=column,
            marginal="box",
            title=f"{column} का डिस्ट्रीब्यूशन",
            template="plotly_white"
        )
    else:
        fig = px.box(
            df,
            y=column,
            title=f"{column} का बॉक्स प्लॉट (Outliers Detection)",
            template="plotly_white"
        )
    fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
    return fig

def plot_categorical_frequency(df: pd.DataFrame, column: str, top_n: int = 10):
    """कैटेगोरिकल कॉलम की टॉप फ़्रीक्वेंसी का बार चार्ट बनाता है"""
    val_counts = df[column].value_counts().head(top_n).reset_index()
    val_counts.columns = [column, "Count"]
    fig = px.bar(
        val_counts,
        x=column,
        y="Count",
        title=f"टॉप {top_n} श्रेणियां: {column}",
        text="Count",
        template="plotly_white"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
    return fig
