from datetime import datetime

def generate_html_report(metrics: dict, stats_df_html: str, missing_html: str) -> str:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>A | Statistical Diagnostic Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; line-height: 1.6; color: #0f172a; max-width: 1000px; margin: 40px auto; padding: 0 20px; }}
    .brand {{ font-size: 32px; font-weight: 900; letter-spacing: -1px; color: #0f172a; display: inline-block; border-bottom: 3px solid #2563eb; padding-bottom: 2px; margin-bottom: 15px; }}
    h1 {{ color: #0f172a; font-size: 24px; margin-top: 10px; margin-bottom: 20px; }}
    h2 {{ color: #1e3a8a; margin-top: 30px; border-bottom: 1px solid #cbd5e1; padding-bottom: 6px; font-size: 18px; }}
    .badge {{ display: inline-block; padding: 4px 10px; background: #eff6ff; color: #1d4ed8; border-radius: 6px; font-weight: 600; font-size: 13px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
    .card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; text-align: center; }}
    .card-val {{ font-size: 24px; font-weight: 700; color: #2563eb; margin-top: 5px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 14px; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 8px 12px; text-align: left; }}
    th {{ background: #f1f5f9; color: #334155; }}
    tr:nth-child(even) {{ background: #f8fafc; }}
    .footer {{ margin-top: 50px; font-size: 12px; color: #94a3b8; text-align: center; border-top: 1px solid #e2e8f0; padding-top: 15px; }}
</style>
</head>
<body>
    <div class="brand">A</div>
    <h1>Statistical Diagnostics & Data Audit</h1>
    <div><span class="badge">Generated: {now_str}</span> | <span class="badge">Engine: A / Core Analytics</span></div>

    <h2>1. Data Dimensions & Quality Health</h2>
    <div class="grid">
        <div class="card"><div>Total Rows</div><div class="card-val">{metrics.get('rows', 0):,}</div></div>
        <div class="card"><div>Total Columns</div><div class="card-val">{metrics.get('columns', 0)}</div></div>
        <div class="card"><div>Duplicate Rows</div><div class="card-val">{metrics.get('duplicate_rows', 0)}</div></div>
        <div class="card"><div>Missing Cells</div><div class="card-val">{metrics.get('missing_percent', 0)}%</div></div>
    </div>

    <h2>2. Missing Value Profiling</h2>
    {missing_html if missing_html else "<p>✓ No missing values detected in the dataset.</p>"}

    <h2>3. Statistical & Distribution Diagnostics</h2>
    {stats_df_html if stats_df_html else "<p>No numerical features found.</p>"}

    <div class="footer">
        A &bull; High-Performance Data Analyst Workbench
    </div>
</body>
</html>"""
    return html_content
