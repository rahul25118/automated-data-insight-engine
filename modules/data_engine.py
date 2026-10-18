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

def detect_outliers_iqr(df: pd.DataFrame, column: str) -> dict:
    """IQR विधि का उपयोग करके आउटलायर्स का पता लगाता है"""
    col_data = df[column].dropna()
    if len(col_data) == 0:
        return {"count": 0, "percent": 0.0, "lower_bound": 0.0, "upper_bound": 0.0, "indices": []}
    
    q1 = col_data.quantile(0.25)
    q3 = col_data.quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
    percent = (len(outliers) / len(df)) * 100
    
    return {
        "count": len(outliers),
        "percent": round(percent, 2),
        "lower_bound": round(lower_bound, 4),
        "upper_bound": round(upper_bound, 4),
        "indices": outliers.index.tolist()
    }

def handle_outliers(df: pd.DataFrame, column: str, method: str) -> pd.DataFrame:
    """आउटलायर्स को कैप (Winsorize) या रो ड्रॉप करके हैंडल करता है"""
    df_handled = df.copy()
    info = detect_outliers_iqr(df_handled, column)
    
    if info["count"] == 0:
        return df_handled
        
    if method == "Cap (Winsorize)":
        df_handled[column] = np.clip(df_handled[column], info["lower_bound"], info["upper_bound"])
    elif method == "Remove Rows":
        df_handled = df_handled.drop(index=info["indices"])
        
    return df_handled

def smart_auto_clean(df: pd.DataFrame):
    """स्वचालित डेटा क्लीनिंग एल्गोरिथ्म (इम्प्यूटेशन, ट्रिमिंग, डुप्लीकेट्स)"""
    df_cleaned = df.copy()
    logs = []
    
    # 1. स्ट्रिंग कॉलम से व्हाइटस्पेस साफ़ करना
    for col in df_cleaned.select_dtypes(include=["object"]).columns:
        df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
    logs.append("✓ सभी टेक्स्ट कॉलम से व्हाइटस्पेस ट्रिम किया गया।")
    
    # 2. डुप्लिकेट पंक्तियाँ हटाना
    dup_count = df_cleaned.duplicated().sum()
    if dup_count > 0:
        df_cleaned = df_cleaned.drop_duplicates()
        logs.append(f"✓ {dup_count} डुप्लिकेट पंक्तियाँ हटाई गईं।")
        
    # 3. मिसिंग वैल्यूज के लिए स्मार्ट इम्प्यूटेशन
    for col in df_cleaned.columns:
        null_count = df_cleaned[col].isnull().sum()
        if null_count > 0:
            if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                skewness = df_cleaned[col].skew()
                if abs(skewness) > 1:
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
                    logs.append(f"✓ '{col}' (न्यूमेरिकल, स्केव्ड): {null_count} मानों को Median से इम्प्यूट किया गया।")
                else:
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
                    logs.append(f"✓ '{col}' (न्यूमेरिकल, नॉर्मल): {null_count} मानों को Mean से इम्प्यूट किया गया।")
            else:
                mode_val = df_cleaned[col].mode()
                if not mode_val.empty:
                    df_cleaned[col] = df_cleaned[col].fillna(mode_val[0])
                    logs.append(f"✓ '{col}' (कैटेगोरिकल): {null_count} मानों को Mode ('{mode_val[0]}') से इम्प्यूट किया गया।")
                    
    return df_cleaned, logs
