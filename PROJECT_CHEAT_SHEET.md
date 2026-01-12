# Project Summary Cheat Sheet: Spare Part Demand Forecasting

> **Quick revision for interviews!**

---

## 🎯 30-Second Pitch

> "I built a **production-ready demand forecasting system** for spare parts using **Prophet + XGBoost** hybrid approach. It achieves **6.2% MAPE**, features an interactive **Streamlit** dashboard with product-level analysis, automated **CI/CD** via GitHub Actions, **data drift monitoring**, and deploys to **Streamlit Cloud for $0/month**."

---

## 📋 PROBACT Summary

| Section | Details |
|---------|---------|
| **Problem** | Unpredictable spare part demand caused stockouts and overstock. |
| **Role** | **End-to-End:** EDA, Feature Engineering, Prophet + XGBoost models, Streamlit dashboard, CI/CD, Drift Monitoring. |
| **Objective Metrics** | MAPE: **6.2%** (Prophet), **6.68%** (XGBoost). MAE: ~465 units. |
| **Business Impact** | Reduced stockouts, optimized inventory levels, cost savings. |
| **Approach** | Hybrid: Prophet (trend/seasonality) + XGBoost (short-term patterns). |
| **Challenges** | Intermittent demand, data quality. Solved with lag features, rolling means, drift monitoring. |
| **Tools** | Python, Prophet, XGBoost, Streamlit, Azure ML (optional), GitHub Actions, Docker. |

---

## 🔄 Technical Flow (Whiteboard)

```
[Raw CSV Data]
       ↓
[Preprocessing] → day_of_week, month, quarter, is_holiday
       ↓
[Feature Engineering] → lag_7, lag_30, rolling_mean_7
       ↓
┌─────────────────────────────────────┐
│         Hybrid Model                │
│  ┌─────────────┐  ┌─────────────┐   │
│  │   Prophet   │  │   XGBoost   │   │
│  │ (Long-term) │  │(Short-term) │   │
│  └─────────────┘  └─────────────┘   │
└─────────────────────────────────────┘
       ↓
[Forecast Output] → MAE, RMSE, MAPE
       ↓
[Streamlit Dashboard] → Visualizations, CSV Export
       ↓
[Optional: Azure ML Endpoint API]
```

---

## 📐 Key Features Engineered

| Feature | Logic | Purpose |
|---------|-------|---------|
| `lag_7, lag_30` | `shift(7/30)` | Capture weekly/monthly patterns |
| `rolling_mean_7` | `rolling(7).mean()` | Smooth short-term fluctuations |
| `day_of_week, month, quarter` | `.dt.dayofweek, .dt.month` | Seasonality indicators |
| `is_holiday` | Holiday lookup | Special event spikes |

---

## 📊 Model Performance

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| **Prophet** | 464.64 | 491.74 | **6.20%** |
| **XGBoost** | 501.58 | 543.66 | 6.68% |

**Recommendation:**
- **Prophet:** Long-term planning (30+ days)
- **XGBoost:** Short-term operations (<14 days)

---

## 🤔 Why Hybrid Model Approach?

**The Core Insight:** No single model is perfect. Each has strengths and weaknesses.

| Model | Strengths | Weaknesses |
|-------|-----------|------------|
| **Prophet** | Trend, Seasonality, Holidays, Missing data | No external features, weak short-term |
| **XGBoost** | External features (lags), non-linear patterns, fast | No auto-seasonality, needs feature engineering |

**The Hybrid Solution:**
```
Prophet → Captures "Big Picture" (Trend + Seasonality)
                  ↓
       Residuals = Actual - Prophet Prediction
                  ↓
XGBoost → Captures "Details" (Short-term via lags/rolling)
                  ↓
Final Forecast = Prophet + XGBoost(Residuals)
```

**Interview Answer (30 sec):**
> "Prophet excels at capturing long-term trends and seasonality automatically, while XGBoost handles short-term patterns using engineered features. By training XGBoost on Prophet's residuals, we get smooth baseline forecasting plus accurate short-term corrections—best of both worlds."

---

## 🚀 Production Features

| Feature | Status | Details |
|---------|--------|---------|
| ✅ CI/CD | GitHub Actions | Auto test, lint, deploy on push |
| ✅ Data Drift Monitoring | Daily 6 AM | Alerts if drift > 0.3 |
| ✅ Auto Retraining | Weekly | Sundays 2 AM |
| ✅ Alerting | Email, Teams, Slack | On drift/failures |
| ✅ Streamlit Dashboard | Live | Product-level analysis, CSV export |
| ✅ Azure ML Endpoint | Optional | `/score` API |

---

## 🔄 CI/CD Pipeline (Detailed)

**Trigger:** On push to `main`/`develop` or Pull Request

```
[Git Push] → [Lint (flake8)] → [Format (black)] → [Test (pytest)] → [Build Package]
```

| Stage | Tool | Purpose |
|-------|------|---------|
| Lint | `flake8` | Syntax errors, undefined vars |
| Format | `black` | Code style consistency |
| Test | `pytest` | Unit tests + coverage |
| Build | `python -m build` | Package artifacts |

---

## 📊 Data Drift Monitoring (Detailed)

**What:** Detect when production data distribution shifts from training data.

**Metrics Used:**
| Metric | What It Measures | Threshold |
|--------|------------------|-----------|
| **K-S Test** | Numerical columns drift | p < 0.05 |
| **Chi-Square** | Categorical columns drift | p < 0.05 |
| **Mean Shift** | Simple mean comparison | > 20% = Alert |

**Flow:**
```
[Streamlit/Notebook] → [Simulate Drift Button] → [Run Check]
                        ↓
              [K-S/Chi-Square Tests]
                        ↓
         [Visualize Distibution Shifts]
```

**Interview Answer:**
> "I implemented custom drift detection using K-S Test (for numerical) and Chi-Square Test (for categorical). This removes heavy dependencies like Evidently and gives me full control over the statistical thresholds (p < 0.05)."

---

## 💰 Cost Architecture

| Component | Free Option | Enterprise Option |
|-----------|-------------|-------------------|
| **Hosting** | Streamlit Cloud ($0) | Azure App Service ($13/mo) |
| **Compute** | Local forecasting | Azure ML compute |
| **Storage** | GitHub | Azure Blob Storage |

---

## ❓ Interview Q&A Quick Reference

**Q: Why Prophet + XGBoost hybrid?**
> Prophet captures trend/seasonality automatically. XGBoost handles short-term patterns with engineered features. Combining gives best of both worlds.

**Q: How did you handle intermittent demand?**
> Used lag features (lag_7, lag_30), rolling means, and WAPE metric (not MAPE) for items with zeros.

**Q: Biggest challenge?**
> Data drift in production. Solved with automated monitoring (custom statistical tests) and retraining triggers.

**Q: How would you scale this?**
> 1. Azure ML endpoints for high concurrency. 2. Parallel training with Dask/Spark. 3. Multi-region deployment.

**Q: What would you do differently?**
> Add external factors (promotions, weather). Explore Deep Learning (N-BEATS).

---

## ⚡ Scalability & Low Latency

**Challenge:** Prophet is slow (~500ms), XGBoost is fast (~5ms).

**Strategy:**

| Technique | How | Benefit |
|-----------|-----|---------|
| **Pre-train Offline** | Train nightly, load models at startup | No training during request |
| **XGBoost for Real-time** | Use XGBoost only for <100ms API calls | 5-10ms latency |
| **Prophet for Batch** | Run Prophet in dashboard/nightly jobs | 500ms acceptable |
| **Caching (LRU/Redis)** | Cache frequent SKU predictions | Instant repeated queries |
| **Auto-Scaling** | Azure ML: 1-10 instances | Handle traffic spikes |
| **Parallel Training** | Dask/Spark for 1000s of SKUs | Scale horizontally |

**Latency Breakdown:**
```
Real-time API: XGBoost only → 5ms
Batch Dashboard: Prophet + XGBoost → 500ms
Cached Request: → <1ms
```

> "XGBoost handles real-time at 5ms. Prophet runs in batch nightly. Models are pre-trained offline. For scaling, I use Azure ML auto-scaling and cache frequent predictions. For 1000s of SKUs, training is parallelized with Dask."

---

## 🔍 Validation & Deployment Strategy

**1. Validation (Before Production):**
*   **Time-Series CV (Backtesting):** "Rolling Origin" split (Train Jan-Jun, Test Jul). No shuffling (leakage prevention).
*   **Hold-Out Set:** Last month kept unseen for final check.
*   **Baseline:** Checked against Naive Forecast (MASE < 1).

**2. Deployment (to Production):**
*   **Containerization:** Docker (runs anywhere).
*   **Blue-Green Deployment:** Spin up new ver., switch traffic if healthy. Instant rollback.
*   **Monitoring:** Custom Drift Detection (K-S/Chi-Square) + Performance tracking.

**Interview Answer:**
> "I used **Time-Series Cross-Validation** (Rolling Origin) to prevent data leakage. For deployment, I **containerized** with Docker and uses a **Blue-Green strategy** for zero-downtime updates, continuously monitoring for **Data Drift**."

---

## 🛠️ Deep Dive Q&A

**Q: How exactly did you integrate Prophet and XGBoost? (Residual Learning)**
> "It's a sequential process:
> 1. **Prophet** predicts the trend/seasonality.
> 2. I calculate the **Residuals** (Actual - Prophet Prediction).
> 3. **XGBoost** is trained specificially to predict these residuals using lag features.
> 4. **Final Forecast = Prophet Prediction + XGBoost Prediction of Residuals**.
> Prophet handles the 'shape', XGBoost fixes the 'errors'."

**Q: How does Blue-Green Deployment ensure zero downtime?**
> "I deploy the new model to an idle **'Green'** environment while users stay on **'Blue'**. After running smoke tests on Green, I switch the load balancer to Green. If anything breaks, I can **instantly switch back** to Blue, ensuring seamless updates and reliability."

**Q: Future Enhancements & Scalability?**
> "Three key areas:
> 1. **Modeling:** Moving to **Probabilistic Forecasting** (Confidence Intervals) for better safety stock planning.
> 2. **Data:** Integrating **External Signals** (Weather, Promotions).
> 3. **Scale:** Using **Spark/Ray** for parallel training of 100k+ SKUs and adding a **Feature Store**."

---

## 🏃 Quick Demo Commands

```bash
# Run Dashboard
streamlit run streamlit_app/app.py

# Run Drift Analysis (Notebook)
# Open notebooks/04_Drift_Detection.ipynb and run all cells
```

---

**Good Luck! 🍀**
