import datetime

def generate_html_report(metrics: dict, ai_summary: str = "", missing_summary_html: str = "") -> str:
    """
    एक पूरी तरह से स्टाइल की गई, प्रिंट और PDF-फ्रेंडली HTML एग्जीक्यूटिव रिपोर्ट तैयार करता है।
    """
    now_str = datetime.datetime.now().strftime("%d %b %Y, %I:%M %p")
    
    ai_section = f"""
    <div class="section">
        <h2>🤖 AI Executive Insights & Recommendations</h2>
        <div class="ai-box">
            {ai_summary.replace(chr(10), '<br>')}
        </div>
    </div>
    """ if ai_summary else """
    <div class="section">
        <h2>🤖 AI Executive Insights</h2>
        <p style="color: #666; font-style: italic;">इस रिपोर्ट को एक्सपोर्ट करते समय कोई AI समरी जनरेट नहीं की गई थी।</p>
    </div>
    """

    missing_section = f"""
    <div class="section">
        <h2>🔍 Missing Values Breakdown</h2>
        {missing_summary_html}
    </div>
    """ if missing_summary_html else ""

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Executive Data Health & EDA Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #1e293b;
            background-color: #f8fafc;
            margin: 0;
            padding: 30px;
        }}
        .container {{
            max-width: 900px;
            margin: auto;
            background: #ffffff;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }}
        .header {{
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            color: #0f172a;
            font-size: 26px;
        }}
        .timestamp {{
            color: #64748b;
            font-size: 14px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: #f1f5f9;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 22px;
            font-weight: 700;
            color: #0284c7;
        }}
        .metric-label {{
            font-size: 13px;
            color: #475569;
            margin-top: 5px;
        }}
        .section {{
            margin-bottom: 35px;
        }}
        .section h2 {{
            font-size: 18px;
            color: #1e293b;
            border-left: 4px solid #0284c7;
            padding-left: 10px;
            margin-bottom: 15px;
        }}
        .ai-box {{
            background: #f8fafc;
            border: 1px solid #cbd5e1;
            padding: 20px;
            border-radius: 8px;
            white-space: pre-wrap;
            font-size: 14px;
            line-height: 1.7;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
            font-size: 14px;
        }}
        th {{
            background-color: #f8fafc;
            font-weight: 600;
            color: #334155;
        }}
        @media print {{
            body {{ background-color: #fff; padding: 0; }}
            .container {{ box-shadow: none; padding: 0; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Executive Data Health & Insights Report</h1>
            <div class="timestamp">Generated on: {now_str} | Powered by Automated Insight Engine</div>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{metrics.get('rows', 0):,}</div>
                <div class="metric-label">Total Rows</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('columns', 0)}</div>
                <div class="metric-label">Total Columns</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('missing_percent', 0)}%</div>
                <div class="metric-label">Missing Cells</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('duplicate_rows', 0)}</div>
                <div class="metric-label">Duplicate Rows</div>
            </div>
        </div>

        {ai_section}
        {missing_section}
    </div>
</body>
</html>
"""
    return html_content
