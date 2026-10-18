import pandas as pd
import numpy as np

def load_data(file):
    """CSV या Excel फ़ाइल लोड करने का फ़ंक्शन"""
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
    """कॉलम-वाइज़ मिसिंग वैल्यूज़ की समरी"""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    summary_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": missing.values,
        "Missing (%)": missing_pct.round(2).values,
        "Data Type": [str(t) for t in df.dtypes.values]
    })
    return summary_df[summary_df["Missing Count"] > 0].sort_values(by="Missing Count", ascending=False)

def impute_missing_values(df: pd.DataFrame, column: str, strategy: str, custom_val=None):
    df_imputed = df.copy()
    is_num = pd.api.types.is_numeric_dtype(df_imputed[column])

    if strategy == "Mean" and is_num:
        df_imputed[column] = df_imputed[column].fillna(df_imputed[column].mean())
    elif strategy == "Median" and is_num:
        df_imputed[column] = df_imputed[column].fillna(df_imputed[column].median())
    elif strategy == "Mode":
        mode_val = df_imputed[column].mode()
        if not mode_val.empty:
            df_imputed[column] = df_imputed[column].fillna(mode_val[0])
    elif strategy == "Constant Value" and custom_val is not None:
        if is_num:
            try:
                custom_val = float(custom_val)
            except ValueError:
                pass
        df_imputed[column] = df_imputed[column].fillna(custom_val)
    elif strategy == "Drop Rows":
        df_imputed = df_imputed.dropna(subset=[column])

    return df_imputed
