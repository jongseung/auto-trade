import json
import os
import logging
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv
import sys
import os
import time

# 상위 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.kis_api import KisAPI

# 로거 설정
logger = logging.getLogger(__name__)


class AccountAPI:
    def __init__(self):
        load_dotenv()
        self.api = KisAPI()
        self.retry_count = 0
        self.max_retries = 3

    def get_account_balance(self):
        """계좌 잔고 조회"""
        try:
            logger.info("KisAPI.get_account_info() 호출 시작")
            account_info = self.api.get_account_info()
            logger.info(f"KisAPI.get_account_info() 반환값: {account_info}")
            if not account_info:
                logger.error("계좌 정보를 가져올 수 없습니다.")
                return None

            logger.info(f"계좌 정보 조회 성공: {account_info}")

            # 계좌 데이터 처리
            account_data = {
                "total_assets": float(account_info.get("총평가금액", "0")),
                "cash": float(
                    account_info.get("매수금액", "0")
                ),  # 임시로 매수금액을 현금으로 사용
                "securities": float(account_info.get("총평가금액", "0"))
                - float(account_info.get("매수금액", "0")),
                "positions": [],
                "pnl": {
                    "total": float(account_info.get("평가손익", "0")),
                    "percentage": float(account_info.get("수익률", "0")),
                },
            }

            logger.info(f"처리된 계좌 데이터: {account_data}")
            return account_data

        except Exception as e:
            logger.error(f"계좌 잔고 조회 중 오류 발생: {str(e)}")
            return None

    def update_account_data(self):
        """계좌 데이터 업데이트 및 저장"""
        try:
            account_data = self.get_account_balance()
            if not account_data:
                logger.error("계좌 데이터를 가져오지 못했습니다.")
                return None

            # 데이터 파일 저장
            os.makedirs("data", exist_ok=True)
            with open("data/account_data.json", "w") as f:
                json.dump(account_data, f, indent=4)

            logger.info("계좌 데이터 업데이트 완료")
            return account_data
        except Exception as e:
            logger.error(f"계좌 데이터 업데이트 중 오류 발생: {str(e)}")
            return None


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    api = AccountAPI()
    data = api.update_account_data()
    print(json.dumps(data, indent=4, ensure_ascii=False))
