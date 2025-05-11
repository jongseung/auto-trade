import threading
import json
import logging
import time
import schedule
from datetime import datetime
from dashboard import TradingDashboard
from notifier import SystemNotifier
from account_api import AccountAPI

# 로거 설정
logger = logging.getLogger(__name__)


class MonitoringSystem:
    def __init__(self):
        self.dashboard = TradingDashboard()
        self.notifier = SystemNotifier()
        self.account_api = AccountAPI()
        self.running = True

    def update_account_data(self):
        """계좌 데이터 업데이트"""
        try:
            account_data = self.account_api.update_account_data()
            if account_data:
                self.dashboard.update_data(
                    {
                        "account": {
                            "total_assets": account_data["total_assets"],
                            "cash": account_data["cash"],
                            "securities": account_data["securities"],
                        },
                        "positions": {"long": 0, "short": 0},
                        "pnl": [
                            {
                                "date": time.strftime("%Y-%m-%d"),
                                "value": account_data["pnl"]["total"],
                            }
                        ],
                        "risk_metrics": {"var": 0, "sharpe": 0, "max_drawdown": 0},
                    }
                )
                logger.info("계좌 데이터 업데이트 성공")
                self.notifier.notify("계좌 데이터 업데이트 성공", "info")
            else:
                logger.error("계좌 데이터 업데이트 실패")
                self.notifier.notify("계좌 데이터 업데이트 실패", "error")
        except Exception as e:
            logger.error(f"계좌 데이터 업데이트 중 오류 발생: {str(e)}")
            self.notifier.notify(
                f"계좌 데이터 업데이트 중 오류 발생: {str(e)}", "error"
            )

    def schedule_job(self):
        schedule.every(1).minutes.do(self.update_account_data)
        while self.running:
            schedule.run_pending()
            time.sleep(1)

    def start(self):
        """모니터링 시스템 시작"""
        logger.info("모니터링 시스템이 시작되었습니다.")
        self.notifier.notify("모니터링 시스템이 시작되었습니다.", "info")
        # 스케줄러 스레드 시작
        scheduler_thread = threading.Thread(target=self.schedule_job, daemon=True)
        scheduler_thread.start()
        # 대시보드 서버는 메인 스레드에서 실행
        self.dashboard.run_server(debug=True)
        self.running = False
        logger.info("모니터링 시스템이 중지되었습니다.")
        self.notifier.notify("모니터링 시스템이 중지되었습니다.", "info")


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # 모니터링 시스템 시작
    system = MonitoringSystem()
    system.start()
