# Spare Part Demand Forecasting - Implementation Plan

## Project Overview
A production-ready demand forecasting system for spare parts across multiple service centers, featuring ML models (Prophet, XGBoost), interactive dashboards, and Azure ML deployment.

---

## Phase 1: Data & Environment Setup

### 1.1 Data Acquisition
- **Source:** Kaggle - [Spare Parts Demand Dataset](https://www.kaggle.com/datasets) or similar industrial dataset
- **Alternative:** UCI ML Repository, synthetic generation if needed
- **Required Fields:** `date`, `part_id`, `service_center`, `demand_quantity`, `category`

### 1.2 Project Structure
```
d:\Antigravity\Spare Part Demand Forecasting\
├── data/
│   ├── raw/                    # Original Kaggle data
│   └── processed/              # Cleaned, feature-engineered data
├── notebooks/
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis
│   ├── 02_Prophet_Model.ipynb  # Prophet implementation
│   └── 03_XGBoost_Model.ipynb  # XGBoost implementation
├── src/
│   ├── data_loader.py          # Data loading utilities
│   ├── preprocessing.py        # Feature engineering
│   ├── models/
│   │   ├── prophet_model.py
│   │   └── xgboost_model.py
│   └── scoring/
│       └── score.py            # Azure ML scoring script
├── streamlit_app/
│   ├── app.py                  # Main Streamlit dashboard
│   ├── pages/
│   │   ├── upload.py           # Data upload page
│   │   ├── forecast.py         # Forecasting page
│   │   └── comparison.py       # Model comparison page
│   └── components/
│       ├── charts.py
│       └── sidebar.py
├── azure_ml/
│   ├── train_pipeline.py       # Azure ML training pipeline
│   ├── deploy_model.py         # Model deployment script
│   └── config.yaml             # Azure ML configuration
├── requirements.txt
├── README.md
└── .env                        # Environment variables
```

---

## Phase 2: Data Exploration & Preprocessing

### 2.1 Jupyter Notebook: EDA
- Load and explore dataset
- Analyze demand patterns (seasonality, trends)
- Handle missing values and outliers
- Visualize demand by service center, part category

### 2.2 Feature Engineering
| Feature | Description |
|---------|-------------|
| `day_of_week` | Weekday encoding |
| `month`, `quarter` | Seasonal features |
| `lag_7`, `lag_30` | Lagged demand features |
| `rolling_mean_7` | 7-day rolling average |
| `is_holiday` | Holiday indicator |

---

## Phase 3: Model Development

### 3.1 Prophet Model
- Auto-detection of seasonality (weekly, yearly)
- Holiday effects integration
- Changepoint detection for trend shifts
- **Use Case:** Long-term forecasting (30-90 days)

### 3.2 XGBoost Model
- Feature-based regression
- Hyperparameter tuning (GridSearchCV)
- Cross-validation with TimeSeriesSplit
- **Use Case:** Short-term forecasting (1-14 days)

### 3.3 Model Comparison
| Metric | Prophet | XGBoost |
|--------|---------|---------|
| MAE | - | - |
| RMSE | - | - |
| MAPE | - | - |

---

## Phase 4: Streamlit Dashboard

### 4.1 Pages
1. **Home:** Overview, KPIs, quick forecast
2. **Data Upload:** CSV upload, data preview
3. **Forecast:** Select model, part, service center → view predictions
4. **Model Comparison:** Side-by-side performance metrics
5. **Settings:** Model parameters, API configuration

### 4.2 Key Features
- Interactive charts (Plotly)
- Download predictions as CSV
- Real-time forecasting
- Multi-service center support

---

## Phase 5: Azure ML API Endpoints

### 5.1 Azure ML Managed Endpoints
Instead of a separate FastAPI service, we'll deploy models as **Azure ML Managed Endpoints**.

| Endpoint | Type | Description |
|----------|------|-------------|
| `/score` | Real-time | Get forecast for a part (REST API) |
| Batch Endpoint | Batch | Process large datasets asynchronously |

### 5.2 Benefits of Azure ML Endpoints
- **No infrastructure management** - Azure handles scaling
- **Built-in monitoring** - Application Insights integration
- **Model versioning** - Easy A/B testing and rollbacks
- **Authentication** - Key-based or Azure AD auth
- **Cost-effective** - Pay only for compute used

---

## Phase 6: Azure ML Deployment

### 6.1 Training Pipeline
- Register dataset in Azure ML
- Train models with MLflow tracking
- Register best model in Model Registry

### 6.2 Deployment
- Deploy as Azure ML Managed Endpoint
- Enable autoscaling
- Monitor with Application Insights

---

## Phase 7: Verification & Documentation

### 7.1 Testing
- Unit tests for data processing
- Integration tests for API
- End-to-end Streamlit testing

### 7.2 Documentation
- README with setup instructions
- API documentation (Swagger)
- User guide for Streamlit app

---

## Phase 8: CI/CD Pipeline (GitHub Actions + Azure DevOps)

### 8.1 Pipeline Architecture
```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌─────────────┐
│  Git Push   │ → │  Run Tests   │ → │  Train Model  │ → │   Deploy    │
└─────────────┘    └──────────────┘    └───────────────┘    └─────────────┘
                          ↓                    ↓
                   ┌──────────────┐    ┌───────────────┐
                   │  Lint/Format │    │ Register Model│
                   └──────────────┘    └───────────────┘
```

### 8.2 CI Pipeline (on every PR/push)
| Step | Action |
|------|--------|
| 1. Checkout | Clone repository |
| 2. Setup Python | Install dependencies |
| 3. Lint | Run flake8, black formatting check |
| 4. Unit Tests | pytest for data processing, models |
| 5. Build | Package model artifacts |

### 8.3 CD Pipeline (on merge to main)
| Step | Action |
|------|--------|
| 1. Train Model | Run Azure ML training pipeline |
| 2. Evaluate | Compare metrics with baseline |
| 3. Register | Save model to Azure ML Model Registry |
| 4. Deploy | Update Azure ML Managed Endpoint |
| 5. Smoke Test | Validate endpoint is responding |

### 8.4 Project Structure Addition
```
├── .github/
│   └── workflows/
│       ├── ci.yml              # Continuous Integration
│       └── cd.yml              # Continuous Deployment
├── azure_pipelines/
│   └── train-deploy.yml        # Azure DevOps pipeline (optional)
```

---

## Phase 9: Data Drift Monitoring

### 9.1 Azure ML Data Drift Monitor
- **Baseline Dataset:** Training data registered in Azure ML
- **Target Dataset:** Production inference data (collected daily/weekly)
- **Monitoring Frequency:** Daily or Weekly

### 9.2 Drift Metrics Tracked
| Metric | Description |
|--------|-------------|
| Feature Drift | Distribution shift in input features |
| Prediction Drift | Change in model output distribution |
| Data Quality | Missing values, outliers, schema changes |

### 9.3 Alerting
- **Azure Monitor Alerts** → Email/Teams notification
- **Threshold:** Trigger alert if drift score > 0.3
- **Dashboard:** Azure ML Studio drift visualization

### 9.4 Implementation
```python
# azure_ml/data_drift_monitor.py
from azure.ai.ml import MLClient
from azure.ai.ml.entities import DataDriftMonitor

monitor = DataDriftMonitor(
    name="demand-drift-monitor",
    baseline_data=baseline_dataset,
    target_data=production_dataset,
    compute_target="cpu-cluster",
    frequency="Day",
    alert_config={"email": ["alerts@company.com"]}
)
```

---

## Phase 10: Automated Model Retraining

### 10.1 Retraining Triggers
| Trigger | Condition |
|---------|-----------|
| **Scheduled** | Weekly/Monthly retrain |
| **Drift-Based** | Data drift exceeds threshold |
| **Performance-Based** | Model accuracy drops below baseline |
| **Manual** | On-demand via Azure ML Studio or API |

### 10.2 Retraining Pipeline Flow
```
┌─────────────────┐
│  Trigger Event  │ (Scheduled / Drift Alert / Manual)
└────────┬────────┘
         ↓
┌─────────────────┐
│  Collect New    │ (From Azure Blob / Database)
│     Data        │
└────────┬────────┘
         ↓
┌─────────────────┐
│  Feature        │ (Run preprocessing.py)
│  Engineering    │
└────────┬────────┘
         ↓
┌─────────────────┐
│  Train Models   │ (Prophet + XGBoost)
└────────┬────────┘
         ↓
┌─────────────────┐
│  Evaluate       │ (Compare with current production model)
└────────┬────────┘
         ↓
    ┌────┴────┐
    │ Better? │
    └────┬────┘
   Yes   │   No
    ↓    └────→ [Keep Current Model]
┌─────────────────┐
│  Register New   │
│     Model       │
└────────┬────────┘
         ↓
┌─────────────────┐
│  Blue-Green     │ (Gradual traffic shift)
│  Deployment     │
└─────────────────┘
```

### 10.3 Data Collection Pipeline
- **Source:** Azure Blob Storage / Azure SQL / Event Hub
- **Frequency:** Daily incremental ingestion
- **Storage:** Azure ML Dataset (versioned)

### 10.4 Project Structure Addition
```
├── azure_ml/
│   ├── data_drift_monitor.py   # Drift monitoring setup
│   ├── retrain_pipeline.py     # Automated retraining
│   └── data_ingestion.py       # Collect new data
├── src/
│   └── monitoring/
│       ├── drift_detector.py
│       └── alert_handler.py
```

---

## Tech Stack Summary

| Component | Technology |
|-----------|------------|
| ML Models | Prophet, XGBoost, Scikit-learn |
| Dashboard | Streamlit, Plotly |
| API | Azure ML Managed Endpoints |
| Cloud | Azure ML, Azure Blob Storage |
| CI/CD | GitHub Actions, Azure DevOps |
| Monitoring | Azure ML Data Drift, Azure Monitor |
| Data | Pandas, NumPy |
| Notebooks | Jupyter |
| Version Control | Git, GitHub |

---

## Timeline Estimate

| Phase | Duration |
|-------|----------|
| Phase 1: Setup | 1-2 hours |
| Phase 2: EDA & Preprocessing | 2-3 hours |
| Phase 3: Model Development | 3-4 hours |
| Phase 4: Streamlit Dashboard | 3-4 hours |
| Phase 5: Azure ML Endpoints | 2 hours |
| Phase 6: Azure Deployment | 2-3 hours |
| Phase 7: Testing & Docs | 1-2 hours |
| Phase 8: CI/CD Pipeline | 2-3 hours |
| Phase 9: Data Drift Monitoring | 2 hours |
| Phase 10: Automated Retraining | 2-3 hours |

**Total: ~22-28 hours**

---

## Next Steps
1. ✅ Review and approve this implementation plan
2. Download/acquire Kaggle dataset
3. Set up project structure and dependencies
4. Begin with EDA notebook
