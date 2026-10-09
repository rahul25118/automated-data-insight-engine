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
    """IQR मेथड से किसी कॉलम के आउटलायर्स डिटेक्ट करता है"""
    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"count": 0, "percent": 0.0, "lower_bound": None, "upper_bound": None}
    
    series = df[column].dropna()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    pct = (len(outliers) / len(series)) * 100 if len(series) > 0 else 0.0
    
    return {
        "count": len(outliers),
        "percent": round(pct, 2),
        "lower_bound": round(lower_bound, 4),
        "upper_bound": round(upper_bound, 4)
    }

def handle_outliers(df: pd.DataFrame, column: str, method: str = "Cap (Winsorize)"):
    """आउटलायर्स को कैप या रिमूव करता है"""
    df_out = df.copy()
    if not pd.api.types.is_numeric_dtype(df_out[column]):
        return df_out
        
    q1 = df_out[column].quantile(0.25)
    q3 = df_out[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    if method == "Cap (Winsorize)":
        df_out[column] = np.clip(df_out[column], lower_bound, upper_bound)
    elif method == "Remove Rows":
        df_out = df_out[(df_out[column] >= lower_bound) & (df_out[column] <= upper_bound)]
        
    return df_out

def smart_auto_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    1-Click Smart Clean:
    - Skewness के आधार पर Mean vs Median से ऑटो-इम्प्यूट
    - Categorical मिसिंग पर Mode
    - डुप्लिकेट्स को हटाना
    - स्ट्रिंग्स को स्ट्रिप करना
    """
    cleaned_df = df.copy()
    actions_taken = []
    
    # 1. Deduplication
    dups_count = cleaned_df.duplicated().sum()
    if dups_count > 0:
        cleaned_df = cleaned_df.drop_duplicates()
        actions_taken.append(f"✓ {dups_count} डुप्लिकेट पंक्तियाँ हटाई गईं।")

    # 2. String Cleaning (Whitespace strip)
    for col in cleaned_df.select_dtypes(include=["object"]).columns:
        cleaned_df[col] = cleaned_df[col].astype(str).str.strip().replace("nan", np.nan)

    # 3. Smart Missing Value Imputation
    for col in cleaned_df.columns:
        if cleaned_df[col].isnull().any():
            missing_num = cleaned_df[col].isnull().sum()
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                skew_val = cleaned_df[col].skew()
                if abs(skew_val) > 1.0:
                    med = cleaned_df[col].median()
                    cleaned_df[col] = cleaned_df[col].fillna(med)
                    actions_taken.append(f"✓ '{col}' (Skewed: {skew_val:.2f}): {missing_num} मिसिंग वैल्यूज को Median ({med:.2f}) से भरा गया।")
                else:
                    avg = cleaned_df[col].mean()
                    cleaned_df[col] = cleaned_df[col].fillna(avg)
                    actions_taken.append(f"✓ '{col}' (Normal: {skew_val:.2f}): {missing_num} मिसिंग वैल्यूज को Mean ({avg:.2f}) से भरा गया।")
            else:
                mode_val = cleaned_df[col].mode()
                if not mode_val.empty:
                    cleaned_df[col] = cleaned_df[col].fillna(mode_val[0])
                    actions_taken.append(f"✓ '{col}' (Categorical): {missing_num} मिसिंग वैल्यूज को Mode ('{mode_val[0]}') से भरा गया।")

    if not actions_taken:
        actions_taken.append("✓ डेटासेट पहले से ही साफ था; किसी सुधार की आवश्यकता नहीं पड़ी।")

    return cleaned_df, actions_taken
