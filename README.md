# 🛒 Spare Part Demand Forecasting

A production-ready demand forecasting system designed to predict spare part inventory needs across service centers. Built with **Streamlit**, **Prophet**, and **XGBoost**, featuring a cost-effective architecture that runs locally or on the cloud.

---

## ✨ Features

### 📊 **Product-Level Analysis (New!)**
- **🏆 Demand Ranking:** Automatically identifies top-performing products
- **📈 Trend Visualization:** Interactive charts showing historical demand per product
- **🔍 Specific Forecasting:** Select individual products or forecast for the entire inventory
- **📉 Demand Distribution:** Visual breakdown of demand across your catalog

### 🤖 **Hybrid Forecasting Engine**
- **Local Mode (Default):** Runs Prophet & XGBoost directly in the app (Zero Cost)
- **Cloud Mode (Optional):** Connects to Azure ML Managed Endpoints for enterprise scaling
- **Adaptive Models:** Uses Prophet for trend/seasonality and XGBoost for short-term pattern matching

### 🚀 **Production Ready**
- **Deployment:** Ready for [Streamlit Cloud](STREAMLIT_DEPLOY.md) (Free Tier compatible)
- **CI/CD:** Automated testing pipeline with GitHub Actions
- **Monitoring:** Data drift detection and automated alerts
- **Interactive UI:** User-friendly dashboard for uploading data and viewing results

---

## 🚦 Quick Start

### 1. Run Locally
```bash
# Clone repository
git clone https://github.com/shan31/spare-part-demand-forecasting.git
cd spare-part-demand-forecasting

# Install dependencies
pip install -r requirements.txt

# Run Dashboard
streamlit run streamlit_app/app.py
```

### 2. Deploy to Cloud (Free)
See our [Streamlit Cloud Deployment Guide](STREAMLIT_DEPLOY.md) to go live in 5 minutes!

---

## 🛠️ Usage Guide

1.  **Upload Data:** Upload a CSV with `date`, `part_id`, and `demand_quantity` columns.
2.  **Analyze:** View the "Top Products" table to see which parts drive your inventory.
3.  **Select:** Choose a specific Product ID from the dropdown to forecast for that item.
4.  **Forecast:** Click "Generate Forecast" using the built-in local models.
5.  **Export:** Download the forecast results as a CSV file.

---

## 📂 Project Structure

```
├── data/               # Processed datasets and upload samples
├── streamlit_app/      # Main Dashboard application
│   └── app.py          # Application entry point
├── src/                # Core forecasting logic
│   ├── forecasting/    # Prophet & XGBoost model wrappers
│   └── monitoring/     # Drift detection & alerting
├── azure_ml/           # Azure configuration (Optional)
├── .github/workflows/  # CI/CD pipelines
└── PRODUCTION_DEPLOYMENT.md # Detailed deployment docs
```

---

## 💰 Cost Optimization

This project is optimized for **Zero Monthly Cost** by default:
- **Hosting:** Free on Streamlit Cloud
- **Compute:** Local forecasting (runs in browser/app container)
- **Deployment:** GitHub Actions (Free tier)

*Note: Azure ML scripts are included `azure_ml/` for users who require enterprise-grade scalable endpoints, but are disabled by default to save costs.*

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
