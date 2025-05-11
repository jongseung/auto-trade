import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime
import requests
from typing import Optional, Dict, Any
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# 로거 설정
logger = logging.getLogger(__name__)


class SystemNotifier:
    def __init__(self):
        self.config = self.load_config()
        self.setup_logging()
        self.setup_slack()
        self.setup_email()

    def setup_logging(self):
        """로깅 시스템 설정"""
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.logger = logging.getLogger("trading_system")
        self.logger.setLevel(logging.INFO)

        # 파일 핸들러 설정
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, "trading_system.log"),
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(file_handler)

        # 콘솔 핸들러 설정
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(console_handler)

    def load_config(self) -> Dict[str, Any]:
        """알림 설정 로드"""
        config_path = "config/notification_config.json"
        default_config = {
            "telegram": {"enabled": False, "bot_token": "", "chat_id": ""},
            "email": {
                "enabled": False,
                "smtp_server": "",
                "smtp_port": 587,
                "sender_email": "",
                "sender_password": "",
                "receiver_email": "",
            },
            "notification_levels": {"info": True, "warning": True, "error": True},
        }

        try:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    return json.load(f)
            else:
                logger.warning(f"설정 파일을 찾을 수 없습니다: {config_path}")
                # 설정 파일 디렉토리 생성
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                # 기본 설정 파일 저장
                with open(config_path, "w") as f:
                    json.dump(default_config, f, indent=4)
                return default_config
        except Exception as e:
            logger.error(f"설정 파일 로드 중 오류 발생: {str(e)}")
            return default_config

    def setup_slack(self):
        """Slack 설정"""
        self.slack_config = self.config.get("slack", {})
        self.slack_enabled = self.slack_config.get("enabled", False)
        if self.slack_enabled:
            self.webhook_url = self.slack_config.get("webhook_url")
            self.channel = self.slack_config.get("channel", "#autotrade")
            self.username = self.slack_config.get("username", "KIS 자동매매 봇")
            self.icon_emoji = self.slack_config.get(
                "icon_emoji", ":chart_with_upwards_trend:"
            )

    def setup_email(self):
        """이메일 설정"""
        self.email_config = self.config.get("email", {})
        self.email_enabled = self.email_config.get("enabled", False)

    def send_slack_message(self, message: str, level: str = "info"):
        """Slack으로 메시지 전송"""
        if not self.slack_enabled or not self.webhook_url:
            return

        try:
            emoji = {"info": "ℹ️", "warning": "⚠️", "error": "🚨", "success": "✅"}.get(
                level, "ℹ️"
            )

            payload = {
                "channel": self.channel,
                "username": self.username,
                "text": f"{emoji} {message}",
                "icon_emoji": self.icon_emoji,
            }

            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

        except Exception as e:
            self.logger.error(f"Slack 메시지 전송 실패: {str(e)}")

    def send_email(self, subject: str, message: str):
        """이메일 전송"""
        if not self.email_enabled:
            return

        try:
            msg = MIMEMultipart()
            msg["From"] = self.email_config["sender"]
            msg["To"] = self.email_config["recipient"]
            msg["Subject"] = subject

            msg.attach(MIMEText(message, "plain"))

            with smtplib.SMTP(
                self.email_config["smtp_server"], self.email_config["smtp_port"]
            ) as server:
                server.starttls()
                server.login(
                    self.email_config["username"], self.email_config["password"]
                )
                server.send_message(msg)

        except Exception as e:
            self.logger.error(f"이메일 전송 실패: {str(e)}")

    def notify(self, message: str, level: str = "info"):
        """알림 전송

        Args:
            message (str): 알림 메시지
            level (str): 알림 레벨 (info, warning, error)
        """
        if not self.config["notification_levels"].get(level, True):
            return

        logger.info(f"알림 전송: [{level.upper()}] {message}")

        # 로깅
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message)

        # Slack 알림
        self.send_slack_message(message, level)

        # 이메일 알림
        self.send_email(f"Trading System {level.upper()}", message)

    def log_trade_event(self, event_type: str, details: Dict[str, Any]):
        """거래 이벤트 로깅"""
        message = f"거래 이벤트: {event_type}\n"
        message += "\n".join(f"{k}: {v}" for k, v in details.items())

        level = "info"
        if event_type in ["손절", "익절", "일일 손실 초과"]:
            level = "warning"

        self.notify(message, level)

    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """에러 로깅"""
        message = f"에러 발생: {str(error)}\n"
        if context:
            message += "\n".join(f"{k}: {v}" for k, v in context.items())

        self.notify(message, "error")
