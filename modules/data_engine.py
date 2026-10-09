import pandas as pd
import numpy as np

def load_data(file):
    """CSV या Excel फाइल लोड करने का फंक्शन"""
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file)
        else:
            return None, "अमान्य फ़ाइल प्रारूप। कृपया CSV या Excel फ़ाइल अपलोड करें।"
        return df, None
    except Exception as e:
        return None, str(e)

def get_basic_metrics(df: pd.DataFrame):
    """डेटा की बेसिक मेट्रिक्स निकालता है"""
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    missing_percent = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
    duplicate_rows = df.duplicated().sum()

    metrics = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "numerical_cols": len(df.select_dtypes(include=[np.number]).columns),
        "categorical_cols": len(df.select_dtypes(include=["object", "category"]).columns),
        "missing_cells": int(missing_cells),
        "missing_percent": round(missing_percent, 2),
        "duplicate_rows": int(duplicate_rows)
    }
    return metrics

def get_missing_summary(df: pd.DataFrame):
    """कॉलम-वाइज मिसिंग वैल्यूज का टेबल तैयार करता है"""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    summary_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": missing.values,
        "Missing (%)": missing_pct.round(2).values,
        "Data Type": [str(t) for t in df.dtypes.values]
    })
    return summary_df[summary_df["Missing Count"] > 0].sort_values(by="Missing Count", ascending=False)
