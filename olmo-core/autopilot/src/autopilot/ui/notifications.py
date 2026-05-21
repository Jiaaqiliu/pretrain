"""Notification system for long-running training campaigns.

Supports multiple channels:
- Slack webhooks
- Email (SMTP)
- Desktop notifications
- Custom webhooks
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from autopilot.utils.logging import get_logger

log = get_logger("ui.notifications")


class NotificationLevel(enum.Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


@dataclass
class NotificationConfig:
    """Configuration for notification channels."""

    slack_webhook_url: Optional[str] = None
    slack_channel: Optional[str] = None
    email_smtp_host: Optional[str] = None
    email_from: Optional[str] = None
    email_to: Optional[List[str]] = None
    webhook_urls: List[str] = field(default_factory=list)
    min_level: NotificationLevel = NotificationLevel.WARNING


class NotificationManager:
    """Manages sending notifications across configured channels."""

    def __init__(self, config: NotificationConfig):
        self._config = config

    def notify(
        self,
        title: str,
        message: str,
        level: NotificationLevel = NotificationLevel.INFO,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send a notification to all configured channels."""
        if level.value < self._config.min_level.value:
            return

        log.info(f"[{level.value.upper()}] {title}: {message}")

        if self._config.slack_webhook_url:
            self._send_slack(title, message, level, metadata)

        for url in self._config.webhook_urls:
            self._send_webhook(url, title, message, level, metadata)

    def training_started(self, campaign_name: str, num_phases: int, gpu_hours: float) -> None:
        self.notify(
            title="Training Campaign Started",
            message=f"Campaign: {campaign_name}\nPhases: {num_phases}\nEstimated: {gpu_hours:.0f} GPU-hours",
            level=NotificationLevel.INFO,
        )

    def phase_completed(self, phase_name: str, results_summary: str) -> None:
        self.notify(
            title=f"Phase Completed: {phase_name}",
            message=results_summary,
            level=NotificationLevel.SUCCESS,
        )

    def anomaly_detected(self, experiment_id: str, anomaly_message: str, severity: str) -> None:
        level = NotificationLevel.ERROR if severity == "critical" else NotificationLevel.WARNING
        self.notify(
            title=f"Anomaly Detected: {experiment_id}",
            message=anomaly_message,
            level=level,
            metadata={"experiment_id": experiment_id, "severity": severity},
        )

    def experiment_stopped(self, experiment_id: str, reason: str) -> None:
        self.notify(
            title=f"Experiment Stopped: {experiment_id}",
            message=f"Reason: {reason}",
            level=NotificationLevel.WARNING,
        )

    def campaign_completed(self, campaign_name: str, best_loss: float, total_hours: float) -> None:
        self.notify(
            title="Campaign Completed!",
            message=(
                f"Campaign: {campaign_name}\n"
                f"Best loss: {best_loss:.4f}\n"
                f"Total compute: {total_hours:.0f} GPU-hours"
            ),
            level=NotificationLevel.SUCCESS,
        )

    def _send_slack(
        self,
        title: str,
        message: str,
        level: NotificationLevel,
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        emoji_map = {
            NotificationLevel.INFO: ":information_source:",
            NotificationLevel.WARNING: ":warning:",
            NotificationLevel.ERROR: ":rotating_light:",
            NotificationLevel.SUCCESS: ":white_check_mark:",
        }

        payload = {
            "text": f"{emoji_map.get(level, '')} *{title}*\n{message}",
            "unfurl_links": False,
        }

        if self._config.slack_channel:
            payload["channel"] = self._config.slack_channel

        try:
            response = httpx.post(
                self._config.slack_webhook_url,
                json=payload,
                timeout=10.0,
            )
            if response.status_code != 200:
                log.warning(f"Slack notification failed: {response.status_code}")
        except Exception as e:
            log.warning(f"Slack notification error: {e}")

    def _send_webhook(
        self,
        url: str,
        title: str,
        message: str,
        level: NotificationLevel,
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        payload = {
            "title": title,
            "message": message,
            "level": level.value,
            "source": "autopilot",
        }
        if metadata:
            payload["metadata"] = metadata

        try:
            httpx.post(url, json=payload, timeout=10.0)
        except Exception as e:
            log.warning(f"Webhook notification error ({url}): {e}")
