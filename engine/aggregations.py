import pandas as pd

def compute_group_aggregations(df: pd.DataFrame) -> dict:
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    aggs = {}
    for cat in cat_cols[:2]:  # Top 2 categoricals
        for num in num_cols[:3]:  # Top 3 numericals
            key = f"{num}_by_{cat}"
            grouped = df.groupby(cat)[num].agg(['count', 'sum', 'mean']).round(2)
            aggs[key] = grouped.to_dict(orient='index')
    return aggs
