"""
Azure ML Scheduled Retraining Pipeline
Automates model retraining on a schedule (daily, weekly, monthly)
"""

import os
from datetime import datetime
from pathlib import Path
import argparse
import logging

from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.entities import (
    Environment,
    Schedule,
    RecurrenceTrigger,
    RecurrencePattern,
    JobSchedule
)
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScheduledRetrainingPipeline:
    """Manages scheduled retraining jobs in Azure ML."""
    
    def __init__(self):
        self.credential = DefaultAzureCredential()
        self.ml_client = MLClient(
            credential=self.credential,
            subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
            resource_group_name=os.getenv("AZURE_RESOURCE_GROUP"),
            workspace_name=os.getenv("AZURE_ML_WORKSPACE")
        )
        logger.info("Connected to Azure ML workspace")
    
    def create_training_job(
        self,
        job_name: str = "demand-forecast-training",
        data_path: str = "azureml://datastores/workspaceblobstore/paths/data/",
        model_output_path: str = "azureml://datastores/workspaceblobstore/paths/models/"
    ):
        """Create a training job definition."""
        
        # Define environment
        env = Environment(
            name="demand-forecast-env",
            conda_file="azure_ml/conda.yml",
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
        )
        
        # Define training command
        training_job = command(
            name=job_name,
            display_name=f"Demand Forecast Training - {datetime.now().strftime('%Y-%m-%d')}",
            description="Automated retraining of Prophet and XGBoost models",
            environment=env,
            code="./src",
            command="""
                python -c "
from models.prophet_model import ProphetForecaster
from models.xgboost_model import XGBoostForecaster
from data_loader import DataLoader
import pickle
from pathlib import Path

# Load data
loader = DataLoader()
df = loader.load_csv('demand_data.csv', subfolder='processed')

# Train Prophet
prophet = ProphetForecaster()
prophet.train(df)
prophet.save('${{outputs.model_output}}/prophet_model.pkl')

# Train XGBoost
xgb = XGBoostForecaster()
xgb.train(df)
xgb.save('${{outputs.model_output}}/xgboost_model.pkl')

print('Training complete!')
"
            """,
            inputs={
                "data_input": Input(type="uri_folder", path=data_path)
            },
            outputs={
                "model_output": Output(type="uri_folder", path=model_output_path)
            },
            compute="cpu-cluster",
            experiment_name="demand-forecast-retraining"
        )
        
        return training_job
    
    def create_schedule(
        self,
        schedule_name: str,
        frequency: str = "week",
        interval: int = 1,
        day_of_week: str = "Sunday",
        hour: int = 2,
        minute: int = 0
    ):
        """
        Create a recurring schedule for the training job.
        
        Args:
            schedule_name: Name for the schedule
            frequency: 'day', 'week', or 'month'
            interval: How often (e.g., every 1 week)
            day_of_week: For weekly schedules
            hour: Hour to run (0-23)
            minute: Minute to run (0-59)
        """
        
        # Create recurrence pattern
        if frequency == "day":
            recurrence = RecurrenceTrigger(
                frequency="day",
                interval=interval,
                schedule=RecurrencePattern(hours=[hour], minutes=[minute])
            )
        elif frequency == "week":
            recurrence = RecurrenceTrigger(
                frequency="week",
                interval=interval,
                schedule=RecurrencePattern(
                    hours=[hour],
                    minutes=[minute],
                    week_days=[day_of_week]
                )
            )
        elif frequency == "month":
            recurrence = RecurrenceTrigger(
                frequency="month",
                interval=interval,
                schedule=RecurrencePattern(
                    hours=[hour],
                    minutes=[minute],
                    month_days=[1]  # First of the month
                )
            )
        else:
            raise ValueError(f"Invalid frequency: {frequency}")
        
        # Create job
        training_job = self.create_training_job()
        
        # Create schedule
        job_schedule = JobSchedule(
            name=schedule_name,
            trigger=recurrence,
            create_job=training_job
        )
        
        return job_schedule
    
    def deploy_schedule(
        self,
        schedule_name: str = "weekly-retraining",
        frequency: str = "week",
        **kwargs
    ):
        """Deploy a schedule to Azure ML."""
        
        schedule = self.create_schedule(
            schedule_name=schedule_name,
            frequency=frequency,
            **kwargs
        )
        
        # Create or update schedule
        deployed_schedule = self.ml_client.schedules.begin_create_or_update(schedule).result()
        
        logger.info(f"Schedule deployed: {deployed_schedule.name}")
        logger.info(f"  Status: {deployed_schedule.provisioning_state}")
        logger.info(f"  Trigger: {frequency}")
        
        return deployed_schedule
    
    def list_schedules(self):
        """List all schedules in the workspace."""
        schedules = list(self.ml_client.schedules.list())
        
        print(f"\nFound {len(schedules)} schedules:")
        for schedule in schedules:
            print(f"  - {schedule.name}: {schedule.provisioning_state}")
        
        return schedules
    
    def disable_schedule(self, schedule_name: str):
        """Disable a schedule."""
        self.ml_client.schedules.begin_disable(name=schedule_name).result()
        logger.info(f"Schedule disabled: {schedule_name}")
    
    def enable_schedule(self, schedule_name: str):
        """Enable a schedule."""
        self.ml_client.schedules.begin_enable(name=schedule_name).result()
        logger.info(f"Schedule enabled: {schedule_name}")
    
    def delete_schedule(self, schedule_name: str):
        """Delete a schedule."""
        self.ml_client.schedules.begin_delete(name=schedule_name).result()
        logger.info(f"Schedule deleted: {schedule_name}")
    
    def run_now(self):
        """Trigger an immediate training run."""
        training_job = self.create_training_job(
            job_name=f"manual-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        
        job = self.ml_client.jobs.create_or_update(training_job)
        logger.info(f"Training job started: {job.name}")
        logger.info(f"  Studio URL: {job.studio_url}")
        
        return job


def main():
    parser = argparse.ArgumentParser(description="Manage scheduled retraining")
    parser.add_argument("action", choices=["create", "list", "enable", "disable", "delete", "run-now"])
    parser.add_argument("--name", default="weekly-retraining", help="Schedule name")
    parser.add_argument("--frequency", default="week", choices=["day", "week", "month"])
    parser.add_argument("--day", default="Sunday", help="Day of week for weekly schedules")
    parser.add_argument("--hour", type=int, default=2, help="Hour to run (0-23)")
    
    args = parser.parse_args()
    
    pipeline = ScheduledRetrainingPipeline()
    
    if args.action == "create":
        pipeline.deploy_schedule(
            schedule_name=args.name,
            frequency=args.frequency,
            day_of_week=args.day,
            hour=args.hour
        )
    elif args.action == "list":
        pipeline.list_schedules()
    elif args.action == "enable":
        pipeline.enable_schedule(args.name)
    elif args.action == "disable":
        pipeline.disable_schedule(args.name)
    elif args.action == "delete":
        pipeline.delete_schedule(args.name)
    elif args.action == "run-now":
        pipeline.run_now()


if __name__ == "__main__":
    main()
