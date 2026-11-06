import numpy as np
import pandas as pd
from scipy import stats

def compute_column_metrics(series: pd.Series) -> dict:
    clean = series.dropna()
    n = len(clean)
    if n == 0:
        return {"count": 0, "null_count": int(series.isna().sum())}

    q1 = float(np.percentile(clean, 25))
    q3 = float(np.percentile(clean, 75))
    iqr = q3 - q1
    mean = float(clean.mean())
    sample_std = float(clean.std(ddof=1)) if n > 1 else 0.0

    return {
        "count": n,
        "null_count": int(series.isna().sum()),
        "mean": round(mean, 4),
        "median": round(float(clean.median()), 4),
        "min": round(float(clean.min()), 4),
        "max": round(float(clean.max()), 4),
        "range": round(float(clean.max() - clean.min()), 4),
        "sample_variance": round(float(clean.var(ddof=1)), 4) if n > 1 else 0.0,
        "population_variance": round(float(clean.var(ddof=0)), 4),
        "sample_std": round(sample_std, 4),
        "population_std": round(float(clean.std(ddof=0)), 4),
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "iqr": round(iqr, 4),
        "cv": round(sample_std / mean, 4) if mean != 0 else 0.0,
        "skewness": round(float(stats.skew(clean)), 4) if n > 2 else 0.0,
        "kurtosis": round(float(stats.kurtosis(clean)), 4) if n > 3 else 0.0,
    }

def run_one_sample_ttest(series: pd.Series, mu0: float = 0.0) -> dict:
    clean = series.dropna()
    n = len(clean)
    if n < 2:
        return {"error": "Sample size must be at least 2 for t-test"}
    t_stat, p_val = stats.ttest_1samp(clean, popmean=mu0)
    return {
        "test": "One-Sample t-Test",
        "sample_size": n,
        "hypothesized_mean": mu0,
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_val), 5),
        "reject_null": bool(p_val < 0.05),
        "interpretation": f"{'Significant difference' if p_val < 0.05 else 'No significant difference'} from mu0={mu0} at alpha=0.05"
    }
