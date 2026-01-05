#!/bin/bash
# Azure ML Cleanup Script
# This script removes the Azure ML endpoint to save costs (~$100/month)
# Your app will use local forecasting in Streamlit instead

echo "=============================================="
echo "Azure ML Endpoint Cleanup"
echo "This will DELETE your Azure ML endpoint"
echo "Cost savings: ~$100/month"
echo "=============================================="
echo ""

# Configuration
ENDPOINT_NAME="spare-part-forecast"
RESOURCE_GROUP="your-resource-group"  # REPLACE with your resource group
WORKSPACE_NAME="your-workspace"       # REPLACE with your workspace name

echo "Endpoint to delete: $ENDPOINT_NAME"
echo "Resource Group: $RESOURCE_GROUP"
echo "Workspace: $WORKSPACE_NAME"
echo ""

# Confirm deletion
read -p "Are you sure you want to delete the endpoint? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Deletion cancelled."
    exit 0
fi

echo ""
echo "Step 1: Deleting Azure ML Online Endpoint..."
az ml online-endpoint delete \
  --name $ENDPOINT_NAME \
  --resource-group $RESOURCE_GROUP \
  --workspace-name $WORKSPACE_NAME \
  --yes \
  --no-wait

if [ $? -eq 0 ]; then
    echo "✅ Endpoint deletion started (will complete in 2-3 minutes)"
else
    echo "❌ Failed to delete endpoint. Check if it exists or credentials are correct."
    exit 1
fi

echo ""
echo "Step 2: Checking remaining deployments..."
az ml online-deployment list \
  --endpoint-name $ENDPOINT_NAME \
  --resource-group $RESOURCE_GROUP \
  --workspace-name $WORKSPACE_NAME 2>/dev/null

echo ""
echo "=============================================="
echo "Cleanup Summary"
echo "=============================================="
echo "✅ Endpoint deletion initiated: $ENDPOINT_NAME"
echo "💰 Monthly savings: ~\$100"
echo "📊 Your Streamlit app now uses local forecasting"
echo ""
echo "What was deleted:"
echo "  - Azure ML Online Endpoint (inference server)"
echo "  - Associated deployments"
echo ""
echo "What remains (minimal cost ~\$2/month):"
echo "  - Azure ML Workspace (FREE)"
echo "  - Storage Account (~\$0.50/month)"
echo "  - Container Registry (~\$1.50/month)"
echo ""
echo "To verify deletion (wait 2-3 minutes):"
echo "az ml online-endpoint list --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME"
echo ""
echo "=============================================="
