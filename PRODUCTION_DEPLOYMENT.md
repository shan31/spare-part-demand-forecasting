# Production Deployment Guide

## 🚀 Complete Production Checklist

### ✅ What's Already Production-Ready

| Component | Status | Details |
|-----------|--------|---------|
| **Azure ML Endpoint** | ✅ Live | https://spare-part-forecast.eastus.inference.ml.azure.com/score |
| **CI/CD Pipeline** | ✅ Passing | Both CI and CD pipelines green |
| **Data Drift Monitoring** | ✅ Ready | `drift_monitor.py` functional |
| **Alerting System** | ✅ Ready | Email, Teams, Slack configured |
| **API Documentation** | ✅ Ready | OpenAPI spec + docs |
| **Product Analysis Dashboard** | ✅ Ready | Streamlit with per-product forecasting |

---

## 📋 Production Deployment Steps

### Step 1: Deploy Streamlit Dashboard

#### Option A: Streamlit Cloud (Recommended - Free & Easy)

```bash
# 1. Push all changes to GitHub (already done ✅)
git push origin main

# 2. Go to https://share.streamlit.io/
# 3. Click "New app"
# 4. Connect GitHub repo: shan31/spare-part-demand-forecasting
# 5. Set main file path: streamlit_app/app.py
# 6. Add secrets in Streamlit Cloud dashboard:
#    - AZURE_ML_ENDPOINT_URL
#    - AZURE_ML_API_KEY
# 7. Click "Deploy"
```

**Your app will be live at:** `https://shan31-spare-part-demand-forecasting.streamlit.app`

#### Option B: Azure App Service

```bash
# 1. Create App Service Plan
az appservice plan create --name spare-part-plan \
  --resource-group your-resource-group \
  --sku B1 --is-linux

# 2. Create Web App
az webapp create --name spare-part-dashboard \
  --resource-group your-resource-group \
  --plan spare-part-plan \
  --runtime "PYTHON:3.10"

# 3. Configure environment variables
az webapp config appsettings set --name spare-part-dashboard \
  --resource-group your-resource-group \
  --settings \
  AZURE_ML_ENDPOINT_URL="https://spare-part-forecast.eastus.inference.ml.azure.com/score" \
  AZURE_ML_API_KEY="your-api-key"

# 4. Deploy from GitHub
az webapp deployment source config --name spare-part-dashboard \
  --resource-group your-resource-group \
  --repo-url https://github.com/shan31/spare-part-demand-forecasting \
  --branch main \
  --manual-integration

# 5. Set startup command
az webapp config set --name spare-part-dashboard \
  --resource-group your-resource-group \
  --startup-file "streamlit run streamlit_app/app.py --server.port=8000 --server.address=0.0.0.0"
```

#### Option C: Docker Container

```dockerfile
# Create Dockerfile in project root
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY streamlit_app/ ./streamlit_app/
COPY src/ ./src/
COPY .env.production .env

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
# Build and run
docker build -t spare-part-dashboard .
docker run -p 8501:8501 --env-file .env.production spare-part-dashboard

# Deploy to Azure Container Registry
az acr create --name sparepartacr --resource-group your-rg --sku Basic
docker tag spare-part-dashboard sparepartacr.azurecr.io/dashboard:latest
docker push sparepartacr.azurecr.io/dashboard:latest
```

---

### Step 2: Configure Production Environment

Create `.env.production` file:

```bash
# Azure ML
AZURE_ML_ENDPOINT_URL=https://spare-part-forecast.eastus.inference.ml.azure.com/score
AZURE_ML_API_KEY=your-production-api-key

# Azure Storage
AZURE_STORAGE_CONNECTION_STRING=your-connection-string

# Alerting
SMTP_SERVER=smtp.office365.com
SMTP_PORT=587
SMTP_USERNAME=alerts@yourcompany.com
SMTP_PASSWORD=your-email-password
ALERT_EMAIL=team@yourcompany.com

# Teams & Slack
TEAMS_WEBHOOK_URL=https://outlook.office.com/webhook/...
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

# Azure ML Workspace
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_RESOURCE_GROUP=your-resource-group
AZURE_ML_WORKSPACE=your-workspace-name
```

---

### Step 3: Setup Scheduled Retraining

```bash
# Create weekly retraining schedule (Sundays at 2 AM)
python azure_ml/scheduled_pipeline.py create \
  --name weekly-retraining \
  --frequency week \
  --day Sunday \
  --hour 2

# Verify schedule
python azure_ml/scheduled_pipeline.py list

# Enable the schedule
python azure_ml/scheduled_pipeline.py enable --name weekly-retraining
```

---

### Step 4: Setup Data Drift Monitoring

Add to crontab (or Azure Logic App):

```bash
# Run drift check daily at 6 AM
0 6 * * * cd /app && python src/monitoring/drift_monitor.py \
  --reference data/processed/baseline.csv \
  --current data/processed/new_data.csv \
  --output drift_results.json
```

Or create Azure Logic App:
- Trigger: Daily at 6 AM
- Action: Run drift monitor script
- Send alert if drift detected

---

### Step 5: Configure Monitoring & Alerts

#### Azure Application Insights

```bash
# Enable Application Insights
az monitor app-insights component create \
  --app spare-part-insights \
  --location eastus \
  --resource-group your-resource-group

# Get instrumentation key
az monitor app-insights component show \
  --app spare-part-insights \
  --resource-group your-resource-group \
  --query instrumentationKey
```

Add to Streamlit app:

```python
# streamlit_app/app.py (add at top)
import logging
from opencensus.ext.azure.log_exporter import AzureLogHandler

logger = logging.getLogger(__name__)
logger.addHandler(AzureLogHandler(
    connection_string='InstrumentationKey=your-key'
))
```

#### Test Alerting System

```bash
# Test email alerts
python -c "
from src.monitoring.alerts import create_default_alert_manager, Alert, AlertType, AlertSeverity
manager = create_default_alert_manager()
alert = Alert(
    alert_type=AlertType.SYSTEM,
    severity=AlertSeverity.INFO,
    title='Production Deployment Test',
    message='Dashboard is now live!'
)
manager.send_alert(alert)
"
```

---

### Step 6: Security Hardening

#### API Key Rotation

```bash
# Rotate Azure ML API keys every 90 days
az ml online-endpoint regenerate-keys \
  --name spare-part-forecast \
  --resource-group your-resource-group \
  --workspace-name your-workspace \
  --key primary
```

#### Enable HTTPS & Authentication

For Streamlit Cloud:
- HTTPS enabled by default ✅

For Azure App Service:
```bash
# Enable HTTPS only
az webapp update --name spare-part-dashboard \
  --resource-group your-resource-group \
  --https-only true

# Enable Azure AD authentication
az webapp auth update --name spare-part-dashboard \
  --resource-group your-resource-group \
  --enabled true \
  --action LoginWithAzureActiveDirectory
```

#### Network Security

```bash
# Restrict Azure ML endpoint to specific IPs
az ml online-endpoint update \
  --name spare-part-forecast \
  --resource-group your-resource-group \
  --workspace-name your-workspace \
  --public-network-access disabled

# Enable private endpoint
az ml online-endpoint create-private-endpoint \
  --name spare-part-forecast-pe \
  --endpoint-name spare-part-forecast \
  --vnet-name your-vnet \
  --subnet-name your-subnet
```

---

### Step 7: Performance Optimization

#### Enable Caching

Add to `streamlit_app/app.py`:

```python
import streamlit as st

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data(file):
    return pd.read_csv(file)

@st.cache_resource
def load_model():
    # Cache model loading
    pass
```

#### Azure ML Auto-scaling

```bash
# Configure auto-scaling
az ml online-deployment update \
  --name blue \
  --endpoint-name spare-part-forecast \
  --resource-group your-resource-group \
  --workspace-name your-workspace \
  --instance-count 1 \
  --max-concurrent-requests-per-instance 10
```

---

### Step 8: Backup & Disaster Recovery

```bash
# Backup trained models to Azure Blob
az storage blob upload-batch \
  --destination models-backup \
  --source models/ \
  --account-name yourstorageaccount

# Schedule daily backups
# Add to crontab or Azure Logic App:
0 0 * * * az storage blob upload-batch \
  --destination models-backup-$(date +\%Y\%m\%d) \
  --source models/
```

---

## 🎯 Production Readiness Verification

Run this checklist before going live:

```bash
# 1. Test Azure ML endpoint
python azure_ml/test_endpoint.py

# 2. Verify all enhancements
python verify_enhancements.py

# 3. Check drift monitoring
python test_drift_monitor.py

# 4. Test dashboard locally
streamlit run streamlit_app/app.py

# 5. Run CI/CD pipeline
git push origin main  # Watch GitHub Actions

# 6. Check security
# - API keys in environment variables ✅
# - No secrets in git ✅
# - HTTPS enabled ✅

# 7. Test alerting
python src/monitoring/alerts.py
```

---

## 📊 Production Monitoring Dashboard

Once deployed, monitor these metrics:

| Metric | Tool | Threshold |
|--------|------|-----------|
| Endpoint Response Time | Azure ML | < 2s |
| Dashboard Load Time | App Insights | < 3s |
| Daily Active Users | Streamlit Analytics | Track |
| Prediction Accuracy | Drift Monitor | > 90% |
| API Error Rate | Azure Monitor | < 1% |
| Model Drift Score | Drift Monitor | < 0.3 |

---

## 🆘 Troubleshooting

### Issue: Dashboard won't start

```bash
# Check logs
streamlit run streamlit_app/app.py --logger.level=debug

# Or in Azure:
az webapp log tail --name spare-part-dashboard --resource-group your-rg
```

### Issue: Azure ML endpoint timeout

```bash
# Increase timeout
# In streamlit_app/app.py, line ~287:
response = requests.post(endpoint_url, json=payload, headers=headers, timeout=60)  # Increase from 30
```

### Issue: High memory usage

```bash
# Add to requirements.txt:
# streamlit-server-state

# Enable server-side caching in config.toml
```

---

## 🎉 Post-Deployment

1. Share dashboard URL with team
2. Setup user training session
3. Create user manual
4. Setup weekly review meetings
5. Plan for future enhancements

---

## 📞 Support & Maintenance

- **GitHub Repository:** https://github.com/shan31/spare-part-demand-forecasting
- **API Documentation:** See `docs/API.md`
- **Model Retraining:** Automated weekly on Sundays at 2 AM
- **Drift Alerts:** Sent to configured email/Teams/Slack
- **CI/CD Status:** https://github.com/shan31/spare-part-demand-forecasting/actions

---

**Last Updated:** January 2026
**Version:** 1.0.0
**Status:** Production Ready ✅
