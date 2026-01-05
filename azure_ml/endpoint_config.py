"""
Azure ML Endpoint Configuration
Manages the Azure ML managed endpoint setup
"""

import os
from pathlib import Path
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration
)
from azure.identity import DefaultAzureCredential
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AzureMLEndpointManager:
    """Manage Azure ML managed endpoints for model deployment."""
    
    def __init__(
        self,
        subscription_id: str = None,
        resource_group: str = None,
        workspace_name: str = None
    ):
        self.subscription_id = subscription_id or os.getenv('AZURE_SUBSCRIPTION_ID')
        self.resource_group = resource_group or os.getenv('AZURE_RESOURCE_GROUP')
        self.workspace_name = workspace_name or os.getenv('AZURE_ML_WORKSPACE')
        
        self.ml_client = None
        self._connect()
    
    def _connect(self):
        """Connect to Azure ML workspace."""
        try:
            credential = DefaultAzureCredential()
            self.ml_client = MLClient(
                credential=credential,
                subscription_id=self.subscription_id,
                resource_group_name=self.resource_group,
                workspace_name=self.workspace_name
            )
            logger.info(f"Connected to Azure ML workspace: {self.workspace_name}")
        except Exception as e:
            logger.error(f"Failed to connect to Azure ML: {e}")
            raise
    
    def create_endpoint(
        self,
        endpoint_name: str = "demand-forecast-endpoint",
        description: str = "Spare Part Demand Forecasting API"
    ) -> ManagedOnlineEndpoint:
        """Create a managed online endpoint."""
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description=description,
            auth_mode="key",  # or "aml_token" for AAD auth
            tags={
                "project": "spare-part-forecasting",
                "model_type": "prophet,xgboost"
            }
        )
        
        logger.info(f"Creating endpoint: {endpoint_name}")
        created_endpoint = self.ml_client.online_endpoints.begin_create_or_update(
            endpoint
        ).result()
        
        logger.info(f"Endpoint created: {created_endpoint.name}")
        return created_endpoint
    
    def register_model(
        self,
        model_name: str,
        model_path: str,
        description: str = ""
    ) -> Model:
        """Register a model in Azure ML."""
        model = Model(
            path=model_path,
            name=model_name,
            description=description,
            type="custom_model"
        )
        
        registered_model = self.ml_client.models.create_or_update(model)
        logger.info(f"Model registered: {registered_model.name} v{registered_model.version}")
        
        return registered_model
    
    def create_deployment(
        self,
        endpoint_name: str,
        deployment_name: str = "blue",
        model_name: str = "demand-forecast-model",
        model_version: str = "1",
        instance_type: str = "Standard_DS2_v2",
        instance_count: int = 1
    ) -> ManagedOnlineDeployment:
        """Create a deployment for the endpoint."""
        
        # Define scoring environment
        env = Environment(
            name="demand-forecast-env",
            conda_file="azure_ml/conda.yml",
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
        )
        
        # Get model reference
        model = self.ml_client.models.get(name=model_name, version=model_version)
        
        deployment = ManagedOnlineDeployment(
            name=deployment_name,
            endpoint_name=endpoint_name,
            model=model,
            environment=env,
            code_configuration=CodeConfiguration(
                code="src/scoring",
                scoring_script="score.py"
            ),
            instance_type=instance_type,
            instance_count=instance_count,
            tags={"version": model_version}
        )
        
        logger.info(f"Creating deployment: {deployment_name}")
        created_deployment = self.ml_client.online_deployments.begin_create_or_update(
            deployment
        ).result()
        
        # Set traffic to 100% for this deployment
        endpoint = self.ml_client.online_endpoints.get(endpoint_name)
        endpoint.traffic = {deployment_name: 100}
        self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        
        logger.info(f"Deployment created and traffic set: {deployment_name}")
        return created_deployment
    
    def get_endpoint_url(self, endpoint_name: str) -> dict:
        """Get the endpoint URL and key."""
        endpoint = self.ml_client.online_endpoints.get(endpoint_name)
        keys = self.ml_client.online_endpoints.get_keys(endpoint_name)
        
        return {
            "scoring_uri": endpoint.scoring_uri,
            "primary_key": keys.primary_key,
            "secondary_key": keys.secondary_key
        }
    
    def delete_endpoint(self, endpoint_name: str):
        """Delete an endpoint."""
        logger.info(f"Deleting endpoint: {endpoint_name}")
        self.ml_client.online_endpoints.begin_delete(endpoint_name).result()
        logger.info(f"Endpoint deleted: {endpoint_name}")


def deploy_models():
    """Deploy Prophet and XGBoost models to Azure ML."""
    manager = AzureMLEndpointManager()
    
    # Create endpoint
    endpoint = manager.create_endpoint(
        endpoint_name="spare-part-forecast",
        description="Spare Part Demand Forecasting API with Prophet and XGBoost models"
    )
    
    # Register models
    manager.register_model(
        model_name="prophet-forecast",
        model_path="models/prophet_model.pkl",
        description="Prophet model for long-term demand forecasting"
    )
    
    manager.register_model(
        model_name="xgboost-forecast",
        model_path="models/xgboost_model.pkl",
        description="XGBoost model for short-term demand forecasting"
    )
    
    # Create deployment
    manager.create_deployment(
        endpoint_name="spare-part-forecast",
        deployment_name="blue",
        model_name="prophet-forecast"
    )
    
    # Get endpoint info
    endpoint_info = manager.get_endpoint_url("spare-part-forecast")
    print(f"\n{'='*50}")
    print("DEPLOYMENT COMPLETE!")
    print(f"{'='*50}")
    print(f"Endpoint URL: {endpoint_info['scoring_uri']}")
    print(f"Primary Key: {endpoint_info['primary_key'][:20]}...")
    print(f"{'='*50}")


if __name__ == "__main__":
    deploy_models()