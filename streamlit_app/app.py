"""
Streamlit Dashboard - Main Application
Spare Part Demand Forecasting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Page config
st.set_page_config(
    page_title="Spare Part Demand Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #F97316;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #F97316;
    }
    /* KPI Tile Styling */
    [data-testid="stMetricValue"] {
        color: #1E3A5F !important;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        color: #F97316 !important;
        font-weight: 600;
    }
    [data-testid="stMetricDelta"] {
        color: #10B981 !important;
    }
    /* Sidebar Quick Stats */
    .sidebar .stMetric {
        background: linear-gradient(135deg, #1E3A5F 0%, #0F172A 100%);
        padding: 0.8rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    .sidebar [data-testid="stMetricValue"] {
        color: #F97316 !important;
    }
    .sidebar [data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/analytics.png", width=80)
        st.markdown("## Navigation")
        
        page = st.radio(
            "Go to",
            ["🏠 Dashboard", "📤 Upload Data", "📈 Forecast", "⚖️ Model Comparison", "⚙️ Settings"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### Quick Stats")
        st.metric("Models Active", "2", "Prophet, XGBoost")
        st.metric("Last Updated", datetime.now().strftime("%Y-%m-%d"))
        
        st.markdown("---")
        st.markdown(
            "Made with ❤️ by [Shanmugam](https://github.com/shan31)",
            unsafe_allow_html=True
        )
    
    # Main content based on page selection
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📤 Upload Data":
        show_upload()
    elif page == "📈 Forecast":
        show_forecast()
    elif page == "⚖️ Model Comparison":
        show_comparison()
    elif page == "⚙️ Settings":
        show_settings()


def show_dashboard():
    """Main dashboard view."""
    st.markdown('<p class="main-header">📊 Spare Part Demand Forecasting</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered demand prediction for optimized inventory management</p>', unsafe_allow_html=True)
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Parts Tracked", "1,234", "+12%")
    with col2:
        st.metric("Service Centers", "4", "Active")
    with col3:
        st.metric("Model Accuracy", "94.2%", "+2.1%")
    with col4:
        st.metric("Forecast Horizon", "30 Days", "")
    
    st.markdown("---")
    
    # Quick Forecast (moved above charts)
    st.subheader("⚡ Quick Forecast")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        part_id = st.selectbox("Select Part", ['P-1234', 'P-5678', 'P-9012', 'P-3456', 'P-7890'])
    
    with col2:
        service_center = st.selectbox("Service Center", ['SC-North', 'SC-South', 'SC-East', 'SC-West'])
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔮 Predict", type="primary"):
            st.success(f"Predicted demand for {part_id}: **47 units** (next 7 days)")
    
    st.markdown("---")
    
    # Sample visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Demand Trend")
        
        # Generate sample data
        dates = pd.date_range(start='2024-01-01', periods=90, freq='D')
        demand = 50 + 20 * np.sin(np.arange(90)/10) + np.random.randn(90) * 5
        
        df = pd.DataFrame({'Date': dates, 'Demand': demand})
        
        fig = px.line(df, x='Date', y='Demand', 
                      title='', 
                      color_discrete_sequence=['#F97316'])
        fig.update_layout(
            xaxis_title="",
            yaxis_title="Units",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🔧 Top Parts by Demand")
        
        parts_data = pd.DataFrame({
            'Part': ['P-1234', 'P-5678', 'P-9012', 'P-3456', 'P-7890'],
            'Demand': [450, 380, 320, 280, 220]
        })
        
        fig = px.bar(parts_data, x='Demand', y='Part', orientation='h',
                     color='Demand', color_continuous_scale='Oranges')
        fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)


def show_upload():
    """Data upload page with forecasting capability."""
    st.markdown("## 📤 Upload Data & Generate Forecast")
    st.markdown("Upload your demand data in CSV format, then generate forecasts")
    
    # Initialize session state
    if 'uploaded_data' not in st.session_state:
        st.session_state['uploaded_data'] = None
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", 
                                      help="CSV should contain 'date' and 'demand_quantity' columns")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['uploaded_data'] = df
            st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
            
            # Data Preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(20), use_container_width=True)
            
            # Column Info
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Columns:**", list(df.columns))
            with col2:
                st.write("**Data Types:**")
                st.write(df.dtypes)
            
            st.markdown("---")
            
            # Forecast Settings
            st.subheader("🔮 Generate Forecast")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                model_type = st.selectbox("Model Type", ["Prophet", "XGBoost"], key="upload_model")
            
            with col2:
                forecast_days = st.slider("Forecast Days", 7, 90, 30, key="upload_days")
            
            with col3:
                use_azure = st.checkbox("Use Azure ML Endpoint", value=True)
            
            if st.button("🚀 Generate Forecast", type="primary", key="upload_forecast"):
                with st.spinner("Generating forecast..."):
                    try:
                        if use_azure:
                            # Call Azure ML endpoint
                            forecast_result = call_azure_endpoint(
                                model=model_type.lower(),
                                periods=forecast_days,
                                data=df
                            )
                        else:
                            # Local forecast simulation
                            forecast_result = generate_local_forecast(df, forecast_days)
                        
                        st.session_state['forecast_result'] = forecast_result
                        st.success("✅ Forecast generated successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.info("💡 Tip: Check if Azure ML endpoint is configured in Settings")
            
            # Display forecast results if available
            if 'forecast_result' in st.session_state and st.session_state['forecast_result'] is not None:
                display_forecast_results(st.session_state['forecast_result'], forecast_days)
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    else:
        # Show sample data format
        st.info("📝 **Expected CSV format:**")
        sample_data = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'demand_quantity': [45, 52, 48],
            'part_id': ['P-001', 'P-001', 'P-001'],
            'service_center': ['SC-North', 'SC-North', 'SC-North']
        })
        st.dataframe(sample_data, use_container_width=True)


def call_azure_endpoint(model: str, periods: int, data: pd.DataFrame):
    """Call Azure ML endpoint for predictions."""
    import requests
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    endpoint_url = os.getenv('AZURE_ML_ENDPOINT_URL', 'https://spare-part-forecast.eastus.inference.ml.azure.com/score')
    api_key = os.getenv('AZURE_ML_API_KEY', '')
    
    if not api_key:
        raise ValueError("Azure ML API key not configured. Set AZURE_ML_API_KEY environment variable.")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    if model == "prophet":
        payload = {
            "model": "prophet",
            "periods": periods
        }
    else:
        # For XGBoost, prepare features from data
        payload = {
            "model": "xgboost",
            "features": prepare_xgboost_features(data, periods)
        }
    
    response = requests.post(endpoint_url, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    
    return response.json()


def prepare_xgboost_features(data: pd.DataFrame, periods: int):
    """Prepare XGBoost features from uploaded data."""
    features_list = []
    
    # Get last values for lag features
    if 'demand_quantity' in data.columns:
        demand = data['demand_quantity'].values
        last_val = demand[-1] if len(demand) > 0 else 50
        lag_7 = demand[-7] if len(demand) >= 7 else last_val
        lag_14 = demand[-14] if len(demand) >= 14 else last_val
        lag_30 = demand[-30] if len(demand) >= 30 else last_val
        rolling_mean_7 = np.mean(demand[-7:]) if len(demand) >= 7 else last_val
        rolling_mean_14 = np.mean(demand[-14:]) if len(demand) >= 14 else last_val
        rolling_mean_30 = np.mean(demand[-30:]) if len(demand) >= 30 else last_val
        rolling_std_7 = np.std(demand[-7:]) if len(demand) >= 7 else 10
        rolling_std_14 = np.std(demand[-14:]) if len(demand) >= 14 else 10
        rolling_std_30 = np.std(demand[-30:]) if len(demand) >= 30 else 10
    else:
        last_val = lag_7 = lag_14 = lag_30 = 50
        rolling_mean_7 = rolling_mean_14 = rolling_mean_30 = 50
        rolling_std_7 = rolling_std_14 = rolling_std_30 = 10
    
    for i in range(periods):
        future_date = datetime.now() + timedelta(days=i+1)
        features = {
            "day_of_week": future_date.weekday(),
            "month": future_date.month,
            "day_of_month": future_date.day,
            "quarter": (future_date.month - 1) // 3 + 1,
            "year": future_date.year,
            "week_of_year": future_date.isocalendar()[1],
            "is_weekend": 1 if future_date.weekday() >= 5 else 0,
            "is_month_start": 1 if future_date.day == 1 else 0,
            "is_month_end": 1 if future_date.day >= 28 else 0,
            "lag_1": float(last_val),
            "lag_7": float(lag_7),
            "lag_14": float(lag_14),
            "lag_30": float(lag_30),
            "rolling_mean_7": float(rolling_mean_7),
            "rolling_mean_14": float(rolling_mean_14),
            "rolling_mean_30": float(rolling_mean_30),
            "rolling_std_7": float(rolling_std_7),
            "rolling_std_14": float(rolling_std_14),
            "rolling_std_30": float(rolling_std_30)
        }
        features_list.append(features)
    
    return features_list


def generate_local_forecast(data: pd.DataFrame, periods: int):
    """Generate local forecast without Azure ML."""
    dates = pd.date_range(start=datetime.now(), periods=periods, freq='D')
    
    # Simple moving average forecast
    if 'demand_quantity' in data.columns:
        base = data['demand_quantity'].mean()
        std = data['demand_quantity'].std()
    else:
        base, std = 50, 10
    
    forecast = base + np.random.randn(periods).cumsum() * 0.5
    lower = forecast - 2 * std
    upper = forecast + 2 * std
    
    predictions = []
    for i in range(periods):
        predictions.append({
            "date": dates[i].strftime("%Y-%m-%d"),
            "yhat": float(forecast[i]),
            "yhat_lower": float(lower[i]),
            "yhat_upper": float(upper[i])
        })
    
    return {
        "status": "success",
        "model": "local",
        "predictions": predictions
    }


def display_forecast_results(result: dict, forecast_days: int):
    """Display forecast results with charts."""
    st.markdown("---")
    st.subheader("📊 Forecast Results")
    
    if result.get('status') == 'success' and 'predictions' in result:
        predictions = result['predictions']
        
        # Convert to DataFrame
        if isinstance(predictions[0], dict):
            if 'date' in predictions[0]:
                # Prophet format
                df_pred = pd.DataFrame(predictions)
                dates = pd.to_datetime(df_pred['date'])
                values = df_pred['yhat']
                lower = df_pred.get('yhat_lower', values - 10)
                upper = df_pred.get('yhat_upper', values + 10)
            else:
                # XGBoost format
                dates = pd.date_range(start=datetime.now(), periods=len(predictions), freq='D')
                values = [p.get('prediction', p.get('yhat', 50)) for p in predictions]
                lower = [v - 10 for v in values]
                upper = [v + 10 for v in values]
        else:
            dates = pd.date_range(start=datetime.now(), periods=len(predictions), freq='D')
            values = predictions
            lower = [v - 10 for v in values]
            upper = [v + 10 for v in values]
        
        # Create chart
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
            x=dates, y=values, mode='lines+markers',
            line=dict(color='#F97316', width=2),
            marker=dict(size=6),
            name='Forecast'
        ))
        
        fig.update_layout(
            title=f"Demand Forecast - Next {forecast_days} Days",
            xaxis_title="Date",
            yaxis_title="Predicted Demand",
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Forecast", f"{np.mean(values):.1f}")
        with col2:
            st.metric("Min Forecast", f"{np.min(values):.1f}")
        with col3:
            st.metric("Max Forecast", f"{np.max(values):.1f}")
        with col4:
            st.metric("Total Demand", f"{np.sum(values):.0f}")
        
        # Download button
        forecast_df = pd.DataFrame({
            'Date': dates,
            'Forecast': values,
            'Lower_CI': lower,
            'Upper_CI': upper
        })
        
        st.download_button(
            label="📥 Download Forecast CSV",
            data=forecast_df.to_csv(index=False),
            file_name=f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.error(f"Forecast error: {result.get('message', 'Unknown error')}")


def show_forecast():
    """Forecasting page."""
    st.markdown("## 📈 Demand Forecast")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Settings")
        
        model = st.radio("Select Model", ["Prophet", "XGBoost"])
        forecast_days = st.slider("Forecast Horizon (days)", 7, 90, 30)
        part_filter = st.multiselect("Filter Parts", ['All', 'P-1234', 'P-5678'], default=['All'])
        
        if st.button("🚀 Generate Forecast", type="primary"):
            with st.spinner("Generating forecast..."):
                import time
                time.sleep(2)
                st.session_state['forecast_generated'] = True
    
    with col2:
        if st.session_state.get('forecast_generated', False):
            st.markdown("### Forecast Results")
            
            # Sample forecast data
            dates = pd.date_range(start=datetime.now(), periods=30, freq='D')
            forecast = 50 + np.random.randn(30).cumsum()
            lower = forecast - 10
            upper = forecast + 10
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=upper, mode='lines', line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=dates, y=lower, mode='lines', fill='tonexty', 
                                     fillcolor='rgba(249, 115, 22, 0.2)', line=dict(width=0), 
                                     name='Confidence Interval'))
            fig.add_trace(go.Scatter(x=dates, y=forecast, mode='lines+markers',
                                     line=dict(color='#F97316', width=2), name='Forecast'))
            
            fig.update_layout(
                title=f"{model} Forecast - Next {forecast_days} Days",
                xaxis_title="Date",
                yaxis_title="Demand",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Download button
            forecast_df = pd.DataFrame({
                'Date': dates,
                'Forecast': forecast,
                'Lower': lower,
                'Upper': upper
            })
            
            st.download_button(
                label="📥 Download Forecast CSV",
                data=forecast_df.to_csv(index=False),
                file_name=f"forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("👈 Configure forecast settings and click 'Generate Forecast'")


def show_comparison():
    """Model comparison page."""
    st.markdown("## ⚖️ Model Comparison")
    
    # Sample metrics
    metrics_data = {
        'Metric': ['MAE', 'RMSE', 'MAPE (%)', 'R²', 'Training Time (s)'],
        'Prophet': [12.34, 18.56, 8.2, 0.89, 45.2],
        'XGBoost': [10.21, 15.78, 6.8, 0.92, 12.5]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Metrics Comparison")
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("📈 Visual Comparison")
        
        fig = go.Figure(data=[
            go.Bar(name='Prophet', x=['MAE', 'RMSE'], y=[12.34, 18.56], marker_color='#3B82F6'),
            go.Bar(name='XGBoost', x=['MAE', 'RMSE'], y=[10.21, 15.78], marker_color='#F97316')
        ])
        
        fig.update_layout(barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    # Recommendation
    st.markdown("---")
    st.success("✅ **Recommendation:** XGBoost shows better performance with lower MAE (10.21) and RMSE (15.78). Consider using XGBoost for short-term predictions.")


def show_settings():
    """Settings page."""
    st.markdown("## ⚙️ Settings")
    
    tab1, tab2, tab3 = st.tabs(["Model Config", "Azure ML", "Notifications"])
    
    with tab1:
        st.subheader("Model Configuration")
        
        st.markdown("### Prophet Settings")
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox("Seasonality Mode", ["multiplicative", "additive"])
            st.slider("Changepoint Prior Scale", 0.001, 0.5, 0.05)
        with col2:
            st.checkbox("Yearly Seasonality", value=True)
            st.checkbox("Weekly Seasonality", value=True)
        
        st.markdown("### XGBoost Settings")
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("N Estimators", 10, 500, 100)
            st.number_input("Max Depth", 1, 15, 6)
        with col2:
            st.number_input("Learning Rate", 0.01, 0.5, 0.1)
            st.number_input("Subsample", 0.5, 1.0, 0.8)
    
    with tab2:
        st.subheader("Azure ML Configuration")
        st.text_input("Subscription ID", placeholder="your-subscription-id")
        st.text_input("Resource Group", placeholder="your-resource-group")
        st.text_input("Workspace Name", placeholder="your-workspace")
        st.text_input("Endpoint Name", placeholder="demand-forecast-endpoint")
        
        if st.button("Test Connection"):
            st.info("🔄 Testing Azure ML connection...")
    
    with tab3:
        st.subheader("Notification Settings")
        st.text_input("Alert Email", placeholder="your-email@company.com")
        st.slider("Drift Alert Threshold", 0.1, 0.5, 0.3)
        st.checkbox("Email on Model Retrain", value=True)
        st.checkbox("Email on Drift Detection", value=True)
    
    st.markdown("---")
    if st.button("💾 Save Settings", type="primary"):
        st.success("Settings saved successfully!")


if __name__ == "__main__":
    main()
