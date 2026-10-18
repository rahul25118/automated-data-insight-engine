import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import numpy as np

def plot_distribution(df: pd.DataFrame, col: str, plot_type: str = "Histogram"):
    """डेटा डिस्ट्रीब्यूशन प्लॉट करता है (Histogram/KDE या Box/Violin)"""
    if plot_type == "Histogram":
        fig = px.histogram(
            df, x=col, marginal="box", 
            title=f"Distribution of {col}", 
            template="plotly_white",
            color_discrete_sequence=["#2563EB"]
        )
    elif plot_type == "Violin Plot":
        fig = px.violin(
            df, y=col, box=True, points="all",
            title=f"Violin & Probability Density of {col}",
            template="plotly_white",
            color_discrete_sequence=["#7C3AED"]
        )
    else:
        fig = px.box(
            df, y=col, points="outliers",
            title=f"Box Plot (Outlier View) of {col}",
            template="plotly_white",
            color_discrete_sequence=["#DC2626"]
        )
    fig.update_layout(height=450)
    return fig

def plot_bivariate_scatter(df: pd.DataFrame, x_col: str, y_col: str, color_col: str = None, add_trendline: bool = True):
    """ट्रेंडलाइन और कैटेगोरिकल Hue के साथ स्कैटर प्लॉट"""
    trend = "ols" if add_trendline else None
    fig = px.scatter(
        df, x=x_col, y=y_col, color=color_col if color_col != "None" else None,
        trendline=trend,
        title=f"{y_col} vs {x_col}" + (f" (Colored by {color_col})" if color_col != "None" else ""),
        template="plotly_white",
        opacity=0.75
    )
    fig.update_layout(height=500)
    return fig

def plot_correlation_heatmap(df: pd.DataFrame, method: str = "pearson"):
    """इंटरैक्टिव कोरिलेशन हीटमैप (Pearson या Spearman)"""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return None
    corr = num_df.corr(method=method).round(2)
    fig = px.imshow(
        corr, text_auto=True, aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title=f"Feature Correlation Matrix ({method.capitalize()})",
        template="plotly_white"
    )
    fig.update_layout(height=550)
    return fig

def plot_categorical_frequency(df: pd.DataFrame, col: str, top_n: int = 15):
    """कैटेगोरिकल बार चार्ट (Top N क्लासेस)"""
    val_counts = df[col].value_counts().head(top_n).reset_index()
    val_counts.columns = [col, "Count"]
    fig = px.bar(
        val_counts, x=col, y="Count", text="Count",
        title=f"Frequency Distribution: {col} (Top {top_n})",
        template="plotly_white",
        color="Count",
        color_continuous_scale="Blues"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=450, xaxis_tickangle=-45)
    return fig

def plot_ecdf(df: pd.DataFrame, col: str):
    """Empirical Cumulative Distribution Function (ECDF)"""
    fig = px.ecdf(
        df, x=col, marginal="rug",
        title=f"Empirical Cumulative Distribution (ECDF) of {col}",
        template="plotly_white",
        color_discrete_sequence=["#059669"]
    )
    fig.update_layout(height=450)
    return fig
