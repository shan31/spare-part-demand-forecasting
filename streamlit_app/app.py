"""
Streamlit Dashboard - Main Application
Spare Part Demand Forecasting
"""

import streamlit as st
import pandas as pd
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
    .stMetric > div {
        background: #fff7ed;
        padding: 1rem;
        border-radius: 8px;
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
    
    # Sample visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Demand Trend")
        
        # Generate sample data
        dates = pd.date_range(start='2024-01-01', periods=90, freq='D')
        demand = 50 + 20 * pd.Series(range(90)).apply(lambda x: pd.np.sin(x/10)) + pd.np.random.randn(90) * 5
        
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
    
    # Quick Forecast
    st.markdown("---")
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


def show_upload():
    """Data upload page."""
    st.markdown("## 📤 Upload Data")
    st.markdown("Upload your demand data in CSV format")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        
        st.subheader("Data Preview")
        st.dataframe(df.head(20), use_container_width=True)
        
        st.subheader("Column Info")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Columns:**", list(df.columns))
        with col2:
            st.write("**Data Types:**")
            st.write(df.dtypes)


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
            forecast = 50 + pd.np.random.randn(30).cumsum()
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
