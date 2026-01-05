"""
Azure ML Training Pipeline
Automated model training in Azure ML
"""

import os
from pathlib import Path
from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.entities import Environment, BuildContext
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Azure ML training pipeline for demand forecasting models."""
    
    def __init__(self):
        self.ml_client = self._get_ml_client()
    
    def _get_ml_client(self):
        """Get Azure ML client."""
        credential = DefaultAzureCredential()
        return MLClient(
            credential=credential,
            subscription_id=os.getenv('AZURE_SUBSCRIPTION_ID'),
            resource_group_name=os.getenv('AZURE_RESOURCE_GROUP'),
            workspace_name=os.getenv('AZURE_ML_WORKSPACE')
        )
    
    def create_training_environment(self):
        """Create training environment."""
        env = Environment(
            name="demand-forecast-training",
            description="Environment for training demand forecasting models",
            conda_file="azure_ml/conda.yml",
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
        )
        
        self.ml_client.environments.create_or_update(env)
        logger.info("Training environment created")
        return env
    
    def create_training_job(
        self,
        experiment_name: str = "demand-forecast-training",
        compute_target: str = "cpu-cluster",
        model_type: str = "prophet"
    ):
        """Create and submit training job."""
        
        # Define the training job
        job = command(
            code="./src",
            command=f"python train_{model_type}.py --data-path ${{inputs.data}} --output-path ${{outputs.model}}",
            inputs={
                "data": Input(
                    type=AssetTypes.URI_FOLDER,
                    path="azureml://datastores/workspaceblobstore/paths/data/processed/"
                )
            },
            outputs={
                "model": Output(
                    type=AssetTypes.URI_FOLDER,
                    path=f"azureml://datastores/workspaceblobstore/paths/models/{model_type}/"
                )
            },
            environment="demand-forecast-training@latest",
            compute=compute_target,
            experiment_name=experiment_name,
            display_name=f"{model_type}-training-{os.environ.get('BUILD_ID', 'local')}"
        )
        
        # Submit job
        submitted_job = self.ml_client.jobs.create_or_update(job)
        logger.info(f"Training job submitted: {submitted_job.name}")
        
        return submitted_job
    
    def run_training_pipeline(self):
        """Run the full training pipeline."""
        # Create environment
        self.create_training_environment()
        
        # Train Prophet model
        prophet_job = self.create_training_job(model_type="prophet")
        self.ml_client.jobs.stream(prophet_job.name)
        
        # Train XGBoost model
        xgboost_job = self.create_training_job(model_type="xgboost")
        self.ml_client.jobs.stream(xgboost_job.name)
        
        logger.info("Training pipeline completed!")


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_training_pipeline()
