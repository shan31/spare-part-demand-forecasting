"""
Alerting System for Demand Forecasting
Sends notifications for drift detection, model performance, and system events
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, asdict
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts."""
    DRIFT_DETECTED = "drift_detected"
    MODEL_PERFORMANCE = "model_performance"
    TRAINING_COMPLETE = "training_complete"
    TRAINING_FAILED = "training_failed"
    ENDPOINT_ERROR = "endpoint_error"
    DATA_QUALITY = "data_quality"
    SYSTEM = "system"


@dataclass
class Alert:
    """Alert data structure."""
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    details: Dict[str, Any]
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return {
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp
        }


class AlertChannel:
    """Base class for alert channels."""
    
    def send(self, alert: Alert) -> bool:
        raise NotImplementedError


class EmailAlertChannel(AlertChannel):
    """Send alerts via email."""
    
    def __init__(
        self,
        smtp_server: str = None,
        smtp_port: int = 587,
        username: str = None,
        password: str = None,
        from_email: str = None,
        to_emails: List[str] = None
    ):
        self.smtp_server = smtp_server or os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = smtp_port
        self.username = username or os.getenv("SMTP_USERNAME")
        self.password = password or os.getenv("SMTP_PASSWORD")
        self.from_email = from_email or os.getenv("ALERT_FROM_EMAIL")
        self.to_emails = to_emails or os.getenv("ALERT_TO_EMAILS", "").split(",")
    
    def send(self, alert: Alert) -> bool:
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[{alert.severity.value.upper()}] {alert.title}"
            msg["From"] = self.from_email
            msg["To"] = ", ".join(self.to_emails)
            
            # Plain text
            text = f"""
{alert.title}
{'=' * 50}

Severity: {alert.severity.value.upper()}
Type: {alert.alert_type.value}
Time: {alert.timestamp}

{alert.message}

Details:
{json.dumps(alert.details, indent=2)}
            """
            
            # HTML
            html = f"""
            <html>
            <body style="font-family: Arial, sans-serif;">
                <div style="background: {'#ff4444' if alert.severity == AlertSeverity.CRITICAL else '#ffaa00' if alert.severity == AlertSeverity.WARNING else '#4CAF50'}; 
                            color: white; padding: 20px; border-radius: 5px;">
                    <h2 style="margin: 0;">{alert.title}</h2>
                    <p style="margin: 5px 0;">Severity: {alert.severity.value.upper()}</p>
                </div>
                <div style="padding: 20px; background: #f5f5f5;">
                    <p><strong>Type:</strong> {alert.alert_type.value}</p>
                    <p><strong>Time:</strong> {alert.timestamp}</p>
                    <p>{alert.message}</p>
                    <h3>Details</h3>
                    <pre style="background: #fff; padding: 10px; border-radius: 5px;">
{json.dumps(alert.details, indent=2)}
                    </pre>
                </div>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(text, "plain"))
            msg.attach(MIMEText(html, "html"))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.from_email, self.to_emails, msg.as_string())
            
            logger.info(f"Email alert sent: {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False


class TeamsAlertChannel(AlertChannel):
    """Send alerts to Microsoft Teams via webhook."""
    
    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url or os.getenv("TEAMS_WEBHOOK_URL")
    
    def send(self, alert: Alert) -> bool:
        try:
            import requests
            
            # Map severity to color
            color_map = {
                AlertSeverity.CRITICAL: "FF0000",
                AlertSeverity.WARNING: "FFAA00",
                AlertSeverity.INFO: "00AA00"
            }
            
            # Teams Adaptive Card
            payload = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": color_map.get(alert.severity, "808080"),
                "summary": alert.title,
                "sections": [{
                    "activityTitle": alert.title,
                    "activitySubtitle": f"Severity: {alert.severity.value.upper()}",
                    "facts": [
                        {"name": "Type", "value": alert.alert_type.value},
                        {"name": "Time", "value": alert.timestamp},
                        {"name": "Message", "value": alert.message}
                    ] + [
                        {"name": k, "value": str(v)}
                        for k, v in alert.details.items()
                    ],
                    "markdown": True
                }]
            }
            
            response = requests.post(self.webhook_url, json=payload)
            response.raise_for_status()
            
            logger.info(f"Teams alert sent: {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Teams alert: {e}")
            return False


class SlackAlertChannel(AlertChannel):
    """Send alerts to Slack via webhook."""
    
    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
    
    def send(self, alert: Alert) -> bool:
        try:
            import requests
            
            # Map severity to emoji
            emoji_map = {
                AlertSeverity.CRITICAL: ":red_circle:",
                AlertSeverity.WARNING: ":warning:",
                AlertSeverity.INFO: ":information_source:"
            }
            
            # Slack Block Kit
            payload = {
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": f"{emoji_map.get(alert.severity, ':bell:')} {alert.title}"
                        }
                    },
                    {
                        "type": "section",
                        "fields": [
                            {"type": "mrkdwn", "text": f"*Severity:*\n{alert.severity.value}"},
                            {"type": "mrkdwn", "text": f"*Type:*\n{alert.alert_type.value}"},
                            {"type": "mrkdwn", "text": f"*Time:*\n{alert.timestamp}"}
                        ]
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": alert.message}
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"```{json.dumps(alert.details, indent=2)}```"
                        }
                    }
                ]
            }
            
            response = requests.post(self.webhook_url, json=payload)
            response.raise_for_status()
            
            logger.info(f"Slack alert sent: {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False


class FileAlertChannel(AlertChannel):
    """Log alerts to a file."""
    
    def __init__(self, log_file: str = "alerts/alerts.jsonl"):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def send(self, alert: Alert) -> bool:
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(alert.to_dict()) + "\n")
            
            logger.info(f"Alert logged to file: {alert.title}")
            return True
        except Exception as e:
            logger.error(f"Failed to log alert: {e}")
            return False


class AlertManager:
    """Manages alert routing and sending."""
    
    def __init__(self):
        self.channels: List[AlertChannel] = []
        self.thresholds = {
            "drift_psi": 0.1,
            "drift_ks": 0.05,
            "mape_warning": 15.0,
            "mape_critical": 25.0
        }
    
    def add_channel(self, channel: AlertChannel):
        """Add an alert channel."""
        self.channels.append(channel)
    
    def send_alert(self, alert: Alert):
        """Send alert to all channels."""
        for channel in self.channels:
            try:
                channel.send(alert)
            except Exception as e:
                logger.error(f"Channel {channel.__class__.__name__} failed: {e}")
    
    def alert_drift_detected(
        self,
        feature: str,
        psi_score: float,
        ks_score: float,
        threshold: float
    ):
        """Send drift detection alert."""
        severity = AlertSeverity.CRITICAL if psi_score > 0.25 else AlertSeverity.WARNING
        
        alert = Alert(
            alert_type=AlertType.DRIFT_DETECTED,
            severity=severity,
            title=f"Data Drift Detected: {feature}",
            message=f"Significant drift detected in feature '{feature}'. Consider retraining the model.",
            details={
                "feature": feature,
                "psi_score": round(psi_score, 4),
                "ks_score": round(ks_score, 4),
                "threshold": threshold
            }
        )
        
        self.send_alert(alert)
    
    def alert_model_performance(
        self,
        model_name: str,
        mae: float,
        mape: float,
        baseline_mape: float
    ):
        """Send model performance alert."""
        if mape > self.thresholds["mape_critical"]:
            severity = AlertSeverity.CRITICAL
        elif mape > self.thresholds["mape_warning"]:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.INFO
        
        alert = Alert(
            alert_type=AlertType.MODEL_PERFORMANCE,
            severity=severity,
            title=f"Model Performance Alert: {model_name}",
            message=f"Model MAPE is {mape:.1f}% (baseline: {baseline_mape:.1f}%)",
            details={
                "model": model_name,
                "mae": round(mae, 2),
                "mape": round(mape, 2),
                "baseline_mape": round(baseline_mape, 2),
                "degradation": round(mape - baseline_mape, 2)
            }
        )
        
        self.send_alert(alert)
    
    def alert_training_complete(
        self,
        model_name: str,
        metrics: Dict[str, float],
        duration_seconds: float
    ):
        """Send training completion alert."""
        alert = Alert(
            alert_type=AlertType.TRAINING_COMPLETE,
            severity=AlertSeverity.INFO,
            title=f"Training Complete: {model_name}",
            message=f"Model {model_name} training completed successfully.",
            details={
                "model": model_name,
                "metrics": metrics,
                "duration_seconds": round(duration_seconds, 1)
            }
        )
        
        self.send_alert(alert)
    
    def alert_training_failed(
        self,
        model_name: str,
        error: str
    ):
        """Send training failure alert."""
        alert = Alert(
            alert_type=AlertType.TRAINING_FAILED,
            severity=AlertSeverity.CRITICAL,
            title=f"Training Failed: {model_name}",
            message=f"Model training failed with error: {error}",
            details={
                "model": model_name,
                "error": error
            }
        )
        
        self.send_alert(alert)


def create_default_alert_manager() -> AlertManager:
    """Create alert manager with default channels based on environment."""
    manager = AlertManager()
    
    # Always log to file
    manager.add_channel(FileAlertChannel())
    
    # Add email if configured
    if os.getenv("SMTP_SERVER"):
        manager.add_channel(EmailAlertChannel())
    
    # Add Teams if configured
    if os.getenv("TEAMS_WEBHOOK_URL"):
        manager.add_channel(TeamsAlertChannel())
    
    # Add Slack if configured
    if os.getenv("SLACK_WEBHOOK_URL"):
        manager.add_channel(SlackAlertChannel())
    
    return manager


if __name__ == "__main__":
    # Test the alerting system
    manager = create_default_alert_manager()
    
    # Test drift alert
    manager.alert_drift_detected(
        feature="demand_quantity",
        psi_score=0.15,
        ks_score=0.08,
        threshold=0.1
    )
    
    # Test performance alert
    manager.alert_model_performance(
        model_name="XGBoost",
        mae=125.5,
        mape=12.3,
        baseline_mape=10.5
    )
    
    print("Alerts sent! Check alerts/alerts.jsonl for logged alerts.")
