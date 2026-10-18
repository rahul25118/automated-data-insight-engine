import pandas as pd
import numpy as np

def load_data(uploaded_file):
    try:
        fname = uploaded_file.name.lower()
        if fname.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif fname.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            return None, "असमर्थित फ़ाइल। केवल CSV या Excel लोड करें।"
        return df, None
    except Exception as e:
        return None, str(e)

def get_basic_metrics(df: pd.DataFrame) -> dict:
    total_cells = df.size
    missing_cells = int(df.isnull().sum().sum())
    missing_percent = round((missing_cells / total_cells * 100), 2) if total_cells > 0 else 0.0

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_cells": missing_cells,
        "missing_percent": missing_percent,
        "duplicate_rows": int(df.duplicated().sum()),
        "numerical_cols": len(df.select_dtypes(include=[np.number]).columns),
        "categorical_cols": len(df.select_dtypes(include=["object", "category"]).columns)
    }

def get_statistical_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """न्यूमेरिकल फ़ीचर्स के लिए Skewness, Kurtosis और डिस्ट्रीब्यूशन शेप प्रोफाइलिंग"""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return pd.DataFrame()
    
    stats_list = []
    for col in num_df.columns:
        s = num_df[col].dropna()
        if len(s) == 0:
            continue
        skew_val = round(float(s.skew()), 3)
        kurt_val = round(float(s.kurt()), 3)
        
        if abs(skew_val) < 0.5:
            dist_shape = "Fairly Symmetrical"
        elif abs(skew_val) <= 1.0:
            dist_shape = "Moderately Skewed"
        else:
            dist_shape = "Highly Skewed"

        stats_list.append({
            "Feature": col,
            "Mean": round(float(s.mean()), 2),
            "Std Dev": round(float(s.std()), 2),
            "Skewness": skew_val,
            "Kurtosis": kurt_val,
            "Distribution Shape": dist_shape,
            "Distinct Values": int(s.nunique())
        })
    return pd.DataFrame(stats_list)

def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    missing_cnt = df.isnull().sum()
    missing_cols = missing_cnt[missing_cnt > 0]
    if missing_cols.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "Feature": missing_cols.index,
        "Missing Count": missing_cols.values,
        "Missing (%)": ((missing_cols.values / len(df)) * 100).round(2),
        "Data Type": [str(df[c].dtype) for c in missing_cols.index]
    }).sort_values(by="Missing Count", ascending=False)

def impute_missing_values(df: pd.DataFrame, col: str, strategy: str, custom_val=None) -> pd.DataFrame:
    df_copy = df.copy()
    if strategy == "Mean":
        df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
    elif strategy == "Median":
        df_copy[col] = df_copy[col].fillna(df_copy[col].median())
    elif strategy == "Mode":
        m = df_copy[col].mode()
        if not m.empty:
            df_copy[col] = df_copy[col].fillna(m.iloc[0])
    elif strategy == "Constant Value" and custom_val is not None:
        df_copy[col] = df_copy[col].fillna(custom_val)
    elif strategy == "Drop Rows":
        df_copy = df_copy.dropna(subset=[col])
    return df_copy

def detect_outliers_iqr(df: pd.DataFrame, column: str) -> dict:
    series = df[column].dropna()
    if len(series) == 0:
        return {"count": 0, "percent": 0.0, "lower_bound": 0.0, "upper_bound": 0.0}
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower_bound = round(q1 - 1.5 * iqr, 4)
    upper_bound = round(q3 + 1.5 * iqr, 4)
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return {
        "count": len(outliers),
        "percent": round((len(outliers) / len(series)) * 100, 2),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound
    }

def handle_outliers(df: pd.DataFrame, column: str, method: str = "Cap (Winsorize)") -> pd.DataFrame:
    df_copy = df.copy()
    bounds = detect_outliers_iqr(df_copy, column)
    if method == "Cap (Winsorize)":
        df_copy[column] = df_copy[column].clip(lower=bounds["lower_bound"], upper=bounds["upper_bound"])
    elif method == "Remove Rows":
        df_copy = df_copy[(df_copy[column] >= bounds["lower_bound"]) & (df_copy[column] <= bounds["upper_bound"])]
    return df_copy

def apply_transformation(df: pd.DataFrame, column: str, transform_type: str) -> tuple:
    df_copy = df.copy()
    s = df_copy[column]
    try:
        if transform_type == "Log (ln(x+1))":
            if (s < -1).any():
                return df, "Error: Values < -1 present, cannot apply Log."
            df_copy[f"{column}_log"] = np.log1p(s)
        elif transform_type == "Square Root":
            if (s < 0).any():
                return df, "Error: Negative values present, cannot apply Square Root."
            df_copy[f"{column}_sqrt"] = np.sqrt(s)
        elif transform_type == "Z-Score Standardization":
            mean, std = s.mean(), s.std()
            if std == 0:
                return df, "Error: Standard deviation is zero."
            df_copy[f"{column}_zscore"] = (s - mean) / std
        elif transform_type == "Min-Max Scaling (0 to 1)":
            min_v, max_v = s.min(), s.max()
            if max_v == min_v:
                return df, "Error: Min and Max values are identical."
            df_copy[f"{column}_minmax"] = (s - min_v) / (max_v - min_v)
        return df_copy, None
    except Exception as e:
        return df, str(e)

def smart_auto_clean(df: pd.DataFrame) -> tuple:
    df_clean = df.copy()
    logs = []
    dups = int(df_clean.duplicated().sum())
    if dups > 0:
        df_clean = df_clean.drop_duplicates()
        logs.append(f"✓ {dups} duplicate rows removed.")

    str_cols = df_clean.select_dtypes(include=["object"]).columns
    for c in str_cols:
        df_clean[c] = df_clean[c].apply(lambda x: x.strip() if isinstance(x, str) else x)

    for c in df_clean.columns:
        missing_count = int(df_clean[c].isnull().sum())
        if missing_count == 0:
            continue
        if pd.api.types.is_numeric_dtype(df_clean[c]):
            skew = df_clean[c].skew()
            if abs(skew) > 1.0:
                val = df_clean[c].median()
                df_clean[c] = df_clean[c].fillna(val)
                logs.append(f"✓ '{c}' (Skewed: {skew:.2f}): Imputed {missing_count} nulls with Median ({val:.2f}).")
            else:
                val = df_clean[c].mean()
                df_clean[c] = df_clean[c].fillna(val)
                logs.append(f"✓ '{c}' (Normal: {skew:.2f}): Imputed {missing_count} nulls with Mean ({val:.2f}).")
        else:
            mode_s = df_clean[c].mode()
            if not mode_s.empty:
                val = mode_s.iloc[0]
                df_clean[c] = df_clean[c].fillna(val)
                logs.append(f"✓ '{c}' (Categorical): Imputed {missing_count} nulls with Mode ('{val}').")

    if not logs:
        logs.append("Dataset was already clean.")
    return df_clean, logs
