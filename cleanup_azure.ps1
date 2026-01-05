# Azure ML Cleanup Script (PowerShell)
# This script removes the Azure ML endpoint to save costs (~$100/month)
# Your app will use local forecasting in Streamlit instead

Write-Host "=" * 50
Write-Host "Azure ML Endpoint Cleanup"
Write-Host "This will DELETE your Azure ML endpoint"
Write-Host "Cost savings: ~`$100/month"
Write-Host "=" * 50
Write-Host ""

# Configuration
$ENDPOINT_NAME = "spare-part-forecast"
$RESOURCE_GROUP = "your-resource-group"  # REPLACE with your resource group
$WORKSPACE_NAME = "your-workspace"       # REPLACE with your workspace name

Write-Host "Endpoint to delete: $ENDPOINT_NAME"
Write-Host "Resource Group: $RESOURCE_GROUP"
Write-Host "Workspace: $WORKSPACE_NAME"
Write-Host ""

# Confirm deletion
$confirm = Read-Host "Are you sure you want to delete the endpoint? (yes/no)"

if ($confirm -ne "yes") {
    Write-Host "Deletion cancelled."
    exit 0
}

Write-Host ""
Write-Host "Step 1: Deleting Azure ML Online Endpoint..."

az ml online-endpoint delete `
  --name $ENDPOINT_NAME `
  --resource-group $RESOURCE_GROUP `
  --workspace-name $WORKSPACE_NAME `
  --yes `
  --no-wait

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Endpoint deletion started (will complete in 2-3 minutes)" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to delete endpoint. Check if it exists or credentials are correct." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Step 2: Checking remaining deployments..."
az ml online-deployment list `
  --endpoint-name $ENDPOINT_NAME `
  --resource-group $RESOURCE_GROUP `
  --workspace-name $WORKSPACE_NAME 2>$null

Write-Host ""
Write-Host "=" * 50
Write-Host "Cleanup Summary"
Write-Host "=" * 50
Write-Host "✅ Endpoint deletion initiated: $ENDPOINT_NAME" -ForegroundColor Green
Write-Host "💰 Monthly savings: ~`$100" -ForegroundColor Green
Write-Host "📊 Your Streamlit app now uses local forecasting" -ForegroundColor Cyan
Write-Host ""
Write-Host "What was deleted:"
Write-Host "  - Azure ML Online Endpoint (inference server)"
Write-Host "  - Associated deployments"
Write-Host ""
Write-Host "What remains (minimal cost ~`$2/month):"
Write-Host "  - Azure ML Workspace (FREE)"
Write-Host "  - Storage Account (~`$0.50/month)"
Write-Host "  - Container Registry (~`$1.50/month)"
Write-Host ""
Write-Host "To verify deletion (wait 2-3 minutes):"
Write-Host "az ml online-endpoint list --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME"
Write-Host ""
Write-Host "=" * 50
