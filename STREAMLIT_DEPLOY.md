# Streamlit Cloud Deployment Guide

## 🚀 Deploy Your Spare Part Demand Forecasting Dashboard

Follow these steps to deploy your app to Streamlit Cloud:

---

## Step 1: Sign In to Streamlit Cloud

Go to: **https://share.streamlit.io/**

![Streamlit Cloud Sign In](file:///C:/Users/pop/.gemini/antigravity/brain/2c4d3db8-908a-4943-b669-16853747ff4e/streamlit_auth_options_1767638265538.png)

Choose **"Continue with GitHub"** (recommended since your code is on GitHub)

---

## Step 2: Authorize GitHub Access

- Click "Continue with GitHub"
- Authorize Streamlit to access your GitHub repositories
- Select the `shan31/spare-part-demand-forecasting` repository

---

## Step 3: Create New App

Once logged in:

1. Click **"New app"** button
2. Fill in the deployment form:

| Field | Value |
|-------|-------|
| **Repository** | `shan31/spare-part-demand-forecasting` |
| **Branch** | `main` |
| **Main file path** | `streamlit_app/app.py` |
| **App URL** | `spare-part-forecast` (or your choice) |

---

## Step 4: Configure Secrets (IMPORTANT!)

Before deploying, click **"Advanced settings..."**

In the **Secrets** section, add:

```toml
AZURE_ML_ENDPOINT_URL = "https://spare-part-forecast.eastus.inference.ml.azure.com/score"
AZURE_ML_API_KEY = "YOUR_ACTUAL_API_KEY_HERE"
```

⚠️ **Replace `YOUR_ACTUAL_API_KEY_HERE` with your real Azure ML API key**

To get your API key:
1. Go to Azure ML Studio
2. Navigate to Endpoints → spare-part-forecast
3. Click "Consume" tab
4. Copy the Primary Key

---

## Step 5: Deploy!

1. Click the **"Deploy!"** button
2. Wait 2-3 minutes for deployment
3. Your app will be live at: `https://spare-part-forecast.streamlit.app`

---

## 📊 What Your Users Will See

Once deployed, users can:

1. **Upload CSV files** with demand data
2. **View product analysis**:
   - Top products by demand ranking
   - Demand distribution chart
   - Historical trends
3. **Select a specific product** to forecast
4. **Generate forecasts** for that product
5. **Download** forecast results as CSV

---

## 🔧 Post-Deployment Configuration

### Optional: Custom Domain

1. In Streamlit Cloud dashboard
2. Go to Settings → Custom Domain
3. Add your domain (e.g., `forecast.yourcompany.com`)

### Optional: Enable Authentication

1. Go to Settings → Sharing
2. Choose:
   - **Public** - Anyone can access
   - **Private** - Only invited users
   - **GitHub Teams** - Specific GitHub org members

---

## 📱 Share Your App

Your app URL: `https://spare-part-forecast.streamlit.app`

Share with your team:
```
🚀 Our new Demand Forecasting Dashboard is live!

📊 Features:
- Product-level demand analysis
- AI-powered forecasting (Prophet & XGBoost)
- Historical trend visualization
- CSV export

🔗 Access: https://spare-part-forecast.streamlit.app

📖 How to use:
1. Upload your demand data CSV
2. View which products are in highest demand
3. Select a product
4. Generate forecast
5. Download results
```

---

## 🐛 Troubleshooting

### App won't start?

Check logs in Streamlit Cloud:
1. Go to your app dashboard
2. Click "Manage app"
3. View logs for errors

### Missing environment variables?

Make sure you added secrets correctly:
```toml
AZURE_ML_ENDPOINT_URL = "https://..."
AZURE_ML_API_KEY = "your-key"
```

### Azure ML endpoint timeout?

In Streamlit Cloud settings:
1. Go to Advanced settings
2. Increase timeout to 60 seconds

---

## 📈 Monitoring

### View Analytics

Streamlit Cloud provides:
- Daily active users
- Page views
- Error rates
- Resource usage

Access via: Dashboard → Analytics

### Set Up Alerts

Configure email alerts for:
- App crashes
- High error rates
- Resource limits

---

## 🔄 Updating Your App

When you push changes to GitHub:

```bash
git add .
git commit -m "Update: feature description"
git push origin main
```

Streamlit Cloud will automatically redeploy! 🎉

---

## ✅ Deployment Checklist

Before going live, verify:

- [ ] Secrets are configured (Azure ML credentials)
- [ ] App loads without errors
- [ ] File upload works
- [ ] Product analysis displays
- [ ] Forecasting generates results
- [ ] Download CSV works
- [ ] Team has access

---

## 🆘 Support

- **Streamlit Docs**: https://docs.streamlit.io/streamlit-community-cloud
- **GitHub Issues**: https://github.com/shan31/spare-part-demand-forecasting/issues
- **API Docs**: See `docs/API.md` in your repo

---

**Happy Deploying! 🚀**

Your production-ready demand forecasting dashboard is just a few clicks away!
