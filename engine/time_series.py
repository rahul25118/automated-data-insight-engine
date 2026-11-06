import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def analyze_trend_walk_forward(series: pd.Series) -> dict:
    clean = series.dropna().reset_index(drop=True)
    n = len(clean)
    if n < 5:
        return {"is_temporal": False, "message": "Need >= 5 data points"}

    x = np.arange(n)
    y = clean.values

    # Full trend slope
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    r2 = 1.0 - (np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)) if np.sum((y - y.mean()) ** 2) != 0 else 0.0

    # Walk-forward cross validation (70% train, 30% test) to prevent temporal leakage
    split = int(n * 0.7)
    train_x, train_y = x[:split], y[:split]
    test_x, test_y = x[split:], y[split:]

    wf_slope, wf_intercept = np.polyfit(train_x, train_y, 1)
    wf_preds = wf_slope * test_x + wf_intercept
    wf_mae = mean_absolute_error(test_y, wf_preds)
    wf_rmse = np.sqrt(mean_squared_error(test_y, wf_preds))

    return {
        "is_temporal": True,
        "slope": round(float(slope), 4),
        "intercept": round(float(intercept), 4),
        "r2_score": round(float(r2), 4),
        "walk_forward_mae": round(float(wf_mae), 4),
        "walk_forward_rmse": round(float(wf_rmse), 4),
        "trend_direction": "Upward" if slope > 0 else "Downward" if slope < 0 else "Flat",
        "next_3_forecast": [round(float(slope * (n + i) + intercept), 4) for i in range(3)]
    }
