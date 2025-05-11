import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import telegram
import json
from typing import Dict, List, Optional
import config

logger = logging.getLogger("auto_trade.alert_manager")


class AlertManager:
    def __init__(self):
        self.config = config.load_config()
        self.telegram_bot = None
        self._initialize_telegram()

    def _initialize_telegram(self):
        """텔레그램 봇 초기화"""
        try:
            if "telegram_token" in self.config and "telegram_chat_id" in self.config:
                self.telegram_bot = telegram.Bot(token=self.config["telegram_token"])
                self.telegram_chat_id = self.config["telegram_chat_id"]
                logger.info("텔레그램 봇이 초기화되었습니다.")
        except Exception as e:
            logger.error(f"텔레그램 봇 초기화 실패: {str(e)}")

    def send_alert(self, message: str, level: str = "INFO"):
        """알림 전송"""
        try:
            # 텔레그램으로 전송
            if self.telegram_bot:
                self.telegram_bot.send_message(
                    chat_id=self.telegram_chat_id, text=f"[{level}] {message}"
                )

            # 이메일로 전송 (중요 알림만)
            if level in ["WARNING", "ERROR", "CRITICAL"]:
                self._send_email(message, level)

            logger.info(f"알림 전송 완료: {message}")

        except Exception as e:
            logger.error(f"알림 전송 실패: {str(e)}")

    def _send_email(self, message: str, level: str):
        """이메일 전송"""
        try:
            if "email" not in self.config:
                return

            email_config = self.config["email"]
            msg = MIMEMultipart()
            msg["From"] = email_config["sender"]
            msg["To"] = email_config["recipient"]
            msg["Subject"] = f"[자동매매] {level} 알림"

            body = f"""
            시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            레벨: {level}
            메시지: {message}
            """

            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(
                email_config["smtp_server"], email_config["smtp_port"]
            ) as server:
                server.starttls()
                server.login(email_config["username"], email_config["password"])
                server.send_message(msg)

        except Exception as e:
            logger.error(f"이메일 전송 실패: {str(e)}")

    def send_trade_alert(self, trade_info: Dict):
        """거래 알림 전송"""
        message = f"""
        거래 실행 알림
        종목: {trade_info['ticker']}
        거래 유형: {trade_info['type']}
        수량: {trade_info['quantity']}
        가격: {trade_info['price']:,.0f}원
        총액: {trade_info['total_amount']:,.0f}원
        """
        self.send_alert(message, "INFO")

    def send_error_alert(self, error_info: Dict):
        """에러 알림 전송"""
        message = f"""
        에러 발생 알림
        발생 시간: {error_info['timestamp']}
        에러 유형: {error_info['type']}
        상세 내용: {error_info['message']}
        """
        self.send_alert(message, "ERROR")

    def send_portfolio_alert(self, portfolio_info: Dict):
        """포트폴리오 상태 알림 전송"""
        message = f"""
        포트폴리오 상태 알림
        총 평가금액: {portfolio_info['total_value']:,.0f}원
        일간 수익률: {portfolio_info['daily_return']:.2%}
        보유 종목 수: {portfolio_info['position_count']}개
        """
        self.send_alert(message, "INFO")
