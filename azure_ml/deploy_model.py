"""
Azure ML Deployment Script
Deploys trained models to managed endpoints
"""

import os
import argparse
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


def get_ml_client():
    """Get Azure ML client."""
    credential = DefaultAzureCredential()
    return MLClient(
        credential=credential,
        subscription_id=os.getenv('AZURE_SUBSCRIPTION_ID'),
        resource_group_name=os.getenv('AZURE_RESOURCE_GROUP'),
        workspace_name=os.getenv('AZURE_ML_WORKSPACE')
    )


def deploy_model(
    ml_client: MLClient,
    endpoint_name: str = "spare-part-forecast",
    deployment_name: str = "blue",
    model_path: str = "models/",
    traffic_percentage: int = 100
):
    """Deploy model to managed endpoint."""
    
    # Check if endpoint exists
    try:
        endpoint = ml_client.online_endpoints.get(endpoint_name)
        logger.info(f"Endpoint exists: {endpoint_name}")
    except Exception:
        # Create new endpoint
        logger.info(f"Creating endpoint: {endpoint_name}")
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description="Spare Part Demand Forecasting API",
            auth_mode="key"
        )
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    
    # Register model
    model = Model(
        path=model_path,
        name="demand-forecast-model",
        description="Combined Prophet and XGBoost models"
    )
    registered_model = ml_client.models.create_or_update(model)
    logger.info(f"Model registered: {registered_model.name}")
    
    # Create environment
    env = Environment(
        name="demand-forecast-env",
        conda_file="azure_ml/conda.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
    )
    
    # Create deployment
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=registered_model,
        environment=env,
        code_configuration=CodeConfiguration(
            code="src/scoring",
            scoring_script="score.py"
        ),
        instance_type="Standard_DS2_v2",
        instance_count=1
    )
    
    logger.info(f"Creating deployment: {deployment_name}")
    ml_client.online_deployments.begin_create_or_update(deployment).result()
    
    # Set traffic
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    endpoint.traffic = {deployment_name: traffic_percentage}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    
    logger.info(f"Deployment complete! Traffic: {traffic_percentage}%")
    
    # Get endpoint info
    endpoint_info = ml_client.online_endpoints.get(endpoint_name)
    logger.info(f"Endpoint URL: {endpoint_info.scoring_uri}")
    
    return endpoint_info


def blue_green_deploy(
    ml_client: MLClient,
    endpoint_name: str,
    new_model_path: str
):
    """Blue-green deployment strategy."""
    
    # Get current deployments
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    current_deployment = list(endpoint.traffic.keys())[0]
    new_deployment = "green" if current_deployment == "blue" else "blue"
    
    logger.info(f"Current: {current_deployment}, New: {new_deployment}")
    
    # Deploy new version
    deploy_model(
        ml_client,
        endpoint_name=endpoint_name,
        deployment_name=new_deployment,
        model_path=new_model_path,
        traffic_percentage=0  # No traffic initially
    )
    
    # Gradual traffic shift
    for percentage in [10, 25, 50, 75, 100]:
        endpoint.traffic = {
            current_deployment: 100 - percentage,
            new_deployment: percentage
        }
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        logger.info(f"Traffic shifted: {new_deployment}={percentage}%")
    
    # Delete old deployment
    ml_client.online_deployments.begin_delete(
        name=current_deployment,
        endpoint_name=endpoint_name
    ).result()
    
    logger.info(f"Blue-green deployment complete!")


def main():
    parser = argparse.ArgumentParser(description="Deploy models to Azure ML")
    parser.add_argument("--endpoint-name", default="spare-part-forecast")
    parser.add_argument("--deployment-name", default="blue")
    parser.add_argument("--model-path", default="models/")
    parser.add_argument("--blue-green", action="store_true")
    
    args = parser.parse_args()
    
    ml_client = get_ml_client()
    
    if args.blue_green:
        blue_green_deploy(ml_client, args.endpoint_name, args.model_path)
    else:
        deploy_model(
            ml_client,
            endpoint_name=args.endpoint_name,
            deployment_name=args.deployment_name,
            model_path=args.model_path
        )


if __name__ == "__main__":
    main()
