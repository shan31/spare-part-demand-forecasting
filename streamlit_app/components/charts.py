"""
Charts and visualization components
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def create_demand_trend_chart(df, date_col='date', value_col='demand_quantity'):
    """Create demand trend line chart."""
    fig = px.line(df, x=date_col, y=value_col,
                  color_discrete_sequence=['#F97316'])
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Demand",
        hovermode='x unified',
        margin=dict(l=20, r=20, t=20, b=20)
    )
    return fig


def create_forecast_chart(dates, forecast, lower, upper, model_name='Prophet'):
    """Create forecast chart with confidence interval."""
    fig = go.Figure()
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=dates, y=upper, mode='lines', 
        line=dict(width=0), showlegend=False, name='Upper'
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=lower, mode='lines', 
        fill='tonexty', fillcolor='rgba(249, 115, 22, 0.2)',
        line=dict(width=0), name='Confidence Interval'
    ))
    
    # Forecast line
    fig.add_trace(go.Scatter(
        x=dates, y=forecast, mode='lines+markers',
        line=dict(color='#F97316', width=2), name='Forecast'
    ))
    
    fig.update_layout(
        title=f"{model_name} Forecast",
        xaxis_title="Date",
        yaxis_title="Demand",
        hovermode='x unified'
    )
    return fig


def create_bar_chart(df, x_col, y_col, title='', horizontal=False, color_scale='Oranges'):
    """Create bar chart."""
    if horizontal:
        fig = px.bar(df, x=y_col, y=x_col, orientation='h',
                     color=y_col, color_continuous_scale=color_scale)
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    else:
        fig = px.bar(df, x=x_col, y=y_col,
                     color=y_col, color_continuous_scale=color_scale)
    
    fig.update_layout(
        title=title,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


def create_pie_chart(df, values_col, names_col, title=''):
    """Create pie chart."""
    fig = px.pie(df, values=values_col, names=names_col,
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(title=title)
    return fig


def create_comparison_chart(prophet_metrics, xgb_metrics):
    """Create model comparison bar chart."""
    fig = go.Figure(data=[
        go.Bar(name='Prophet', 
               x=['MAE', 'RMSE', 'MAPE (%)'],
               y=[prophet_metrics['mae'], prophet_metrics['rmse'], prophet_metrics['mape']],
               marker_color='#3B82F6'),
        go.Bar(name='XGBoost',
               x=['MAE', 'RMSE', 'MAPE (%)'],
               y=[xgb_metrics['mae'], xgb_metrics['rmse'], xgb_metrics['mape']],
               marker_color='#F97316')
    ])
    
    fig.update_layout(
        title='Model Comparison: Prophet vs XGBoost',
        barmode='group',
        yaxis_title='Error Value'
    )
    return fig


def create_heatmap(df, x_col, y_col, value_col, title=''):
    """Create heatmap."""
    pivot_df = df.pivot_table(values=value_col, index=y_col, columns=x_col, aggfunc='sum')
    
    fig = px.imshow(pivot_df, text_auto=True, color_continuous_scale='Oranges',
                    labels={'color': value_col})
    fig.update_layout(title=title)
    return fig
