import os
import uuid
import logging
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from typing import Dict, Any, List, Optional

from app.tools.ingester import load_dataset

def generate_visualizations(
    dataset_path: str,
    target_col: Optional[str] = None,
    workspace_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    run_id: str = "default"
) -> Dict[str, Any]:
    """
    Generates interactive Plotly HTML charts with workspace directory support.
    """
    df = load_dataset(dataset_path)
        
    if len(df) > 10000:
        df = df.sample(n=10000, random_state=42)
        
    if workspace_dir:
        final_output_dir = os.path.join(workspace_dir, "plots")
    else:
        final_output_dir = output_dir if output_dir else "sandbox/plots"
        
    os.makedirs(final_output_dir, exist_ok=True)
    chart_paths = []
    
    numeric_cols = [c for c in df.select_dtypes(include='number').columns if not c.endswith('_was_missing')]
    non_numeric_cols = list(df.select_dtypes(exclude='number').columns)
    
    date_col = None
    for col in non_numeric_cols:
        try:
            parsed = pd.to_datetime(df[col], errors='coerce')
            if parsed.notna().mean() > 0.5:
                date_col = col
                df[date_col] = parsed
                break
        except Exception as e:
            logging.warning(f"[run_id={run_id}] Date parse attempt failed for '{col}': {e}")
            
    # Chart 1: Correlation Heatmap
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        fig1 = px.imshow(
            corr,
            text_auto='.2f',
            title='Interactive Feature Correlation Heatmap',
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        p1 = os.path.join(final_output_dir, f"correlation_heatmap_{uuid.uuid4().hex[:6]}.html").replace('\\', '/')
        pio.write_html(fig1, file=p1, include_plotlyjs='cdn')
        chart_paths.append(p1)
        
    # Chart 2: Target / Primary Column Distribution
    primary_col = target_col if target_col and target_col in df.columns else (numeric_cols[0] if numeric_cols else (non_numeric_cols[0] if non_numeric_cols else None))
    if primary_col:
        if primary_col in numeric_cols:
            fig2 = px.histogram(df, x=primary_col, nbins=30, title=f'Distribution of {primary_col}', marginal='rug')
        else:
            counts = df[primary_col].value_counts().reset_index()
            counts.columns = [primary_col, 'count']
            fig2 = px.bar(counts, x=primary_col, y='count', title=f'Distribution of {primary_col}')
        p2 = os.path.join(final_output_dir, f"target_distribution_{uuid.uuid4().hex[:6]}.html").replace('\\', '/')
        pio.write_html(fig2, file=p2, include_plotlyjs='cdn')
        chart_paths.append(p2)
        
    # Chart 3: Outlier Box Plot
    if numeric_cols:
        box_cols = numeric_cols[:4]
        fig3 = px.box(df, y=box_cols, title='Outlier & Distribution Analysis (Box Plots)')
        p3 = os.path.join(final_output_dir, f"outlier_boxplots_{uuid.uuid4().hex[:6]}.html").replace('\\', '/')
        pio.write_html(fig3, file=p3, include_plotlyjs='cdn')
        chart_paths.append(p3)
        
    # Chart 4: Scatter Plot for Highest Correlated Pair
    if len(numeric_cols) >= 2:
        corr_abs = df[numeric_cols].corr().abs()
        max_corr = -1
        pair = (numeric_cols[0], numeric_cols[1])
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                c1, c2 = numeric_cols[i], numeric_cols[j]
                val = corr_abs.loc[c1, c2]
                if not np.isnan(val) and val > max_corr:
                    max_corr = val
                    pair = (c1, c2)
                    
        fig4 = px.scatter(df, x=pair[0], y=pair[1], title=f'Scatter Plot: {pair[0]} vs {pair[1]} (r={max_corr:.2f})')
        p4 = os.path.join(final_output_dir, f"scatter_top_corr_{uuid.uuid4().hex[:6]}.html").replace('\\', '/')
        pio.write_html(fig4, file=p4, include_plotlyjs='cdn')
        chart_paths.append(p4)
        
    # Chart 5: Time Series Line Chart or Categorical Distribution Fallback
    if date_col and numeric_cols:
        temp_df = df.sort_values(date_col)
        fig5 = px.line(temp_df, x=date_col, y=numeric_cols[0], title=f'Time Series Trend: {numeric_cols[0]} over Time')
        p5 = os.path.join(final_output_dir, f"timeseries_trend_{uuid.uuid4().hex[:6]}.html").replace('\\', '/')
        pio.write_html(fig5, file=p5, include_plotlyjs='cdn')
        chart_paths.append(p5)
    elif non_numeric_cols:
        cat_c = non_numeric_cols[0]
        counts = df[cat_c].value_counts().reset_index()
        counts.columns = [cat_c, 'count']
        fig5 = px.bar(counts, x=cat_c, y='count', title=f'Categorical Frequency: {cat_c}')
        p5 = os.path.join(final_output_dir, f"categorical_bar_{uuid.uuid4().hex[:6]}.html").replace('\\', '/')
        pio.write_html(fig5, file=p5, include_plotlyjs='cdn')
        chart_paths.append(p5)

    chart_paths = chart_paths[:5]
    
    summary_lines = [
        "=== VISUALIZATION REPORT ===",
        f"[run_id={run_id}] Generated {len(chart_paths)} Interactive Plotly HTML Chart(s):",
    ]
    for p in chart_paths:
        summary_lines.append(f"  - {p}")
    summary_lines.append("=== VISUALIZATION COMPLETE ===")
    summary_text = "\n".join(summary_lines)
    
    return {
        "chart_paths": chart_paths,
        "charts_generated_count": len(chart_paths),
        "summary_text": summary_text
    }

# Backward compatibility alias
run_visualizations = generate_visualizations
