import numpy as np
import pandas as pd

def detect_iqr_outliers(series: pd.Series, threshold: float = 1.5) -> dict:
    clean = series.dropna()
    if len(clean) < 4:
        return {"outliers": [], "count": 0}
    q1 = np.percentile(clean, 25)
    q3 = np.percentile(clean, 75)
    iqr = q3 - q1
    lower_bound = q1 - (threshold * iqr)
    upper_bound = q3 + (threshold * iqr)
    
    outlier_mask = (series < lower_bound) | (series > upper_bound)
    outlier_indices = series[outlier_mask].index.tolist()
    outlier_values = series[outlier_mask].tolist()

    return {
        "method": "IQR",
        "threshold": threshold,
        "bounds": (round(float(lower_bound), 4), round(round(float(upper_bound), 4))),
        "count": len(outlier_indices),
        "indices": outlier_indices,
        "values": outlier_values
    }

def detect_modified_zscore(series: pd.Series, threshold: float = 3.5) -> dict:
    clean = series.dropna()
    if len(clean) < 4:
        return {"outliers": [], "count": 0}
    median = clean.median()
    mad = (clean - median).abs().median()
    if mad == 0:
        return {"outliers": [], "count": 0}
    
    mod_z = 0.6745 * (series - median).abs() / mad
    outlier_mask = mod_z > threshold
    return {
        "method": "Modified Z-Score (MAD)",
        "count": int(outlier_mask.sum()),
        "indices": series[outlier_mask].index.tolist(),
        "values": series[outlier_mask].tolist()
    }
