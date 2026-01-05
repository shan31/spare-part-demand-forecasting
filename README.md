# Spare Part Demand Forecasting

A production-ready demand forecasting system for spare parts across multiple service centers, featuring ML models (Prophet, XGBoost), interactive Streamlit dashboards, and Azure ML deployment.

## Features

- 📊 **Interactive Dashboard** - Streamlit-based UI for forecasting and visualization
- 🤖 **ML Models** - Prophet for long-term, XGBoost for short-term forecasting
- ☁️ **Azure ML** - Managed endpoints with autoscaling
- 🔄 **CI/CD** - Automated training and deployment pipelines
- 📈 **Data Drift Monitoring** - Automated alerts and retraining triggers

## Quick Start

```bash
# Clone repository
git clone https://github.com/shan31/spare-part-demand-forecasting.git
cd spare-part-demand-forecasting

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_app/app.py
```

## Project Structure

```
├── data/               # Raw and processed datasets
├── notebooks/          # Jupyter notebooks for EDA and modeling
├── src/                # Source code modules
├── streamlit_app/      # Streamlit dashboard
├── azure_ml/           # Azure ML pipelines and configs
├── .github/workflows/  # CI/CD pipelines
└── tests/              # Unit and integration tests
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| ML Models | Prophet, XGBoost |
| Dashboard | Streamlit, Plotly |
| Cloud | Azure ML |
| CI/CD | GitHub Actions |

## License

MIT License
