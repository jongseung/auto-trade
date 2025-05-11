import os
from dotenv import load_dotenv
import requests
import json
from datetime import datetime, timedelta
import time
import jwt
import uuid

# .env 파일 로드
load_dotenv()


class AccountInfoTest:
    def __init__(self):
        self.app_key = os.getenv("APP_KEY")
        self.app_secret = os.getenv("APP_SECRET")
        self.account_number = os.getenv("ACCOUNT_NUMBER")
        self.base_url = "https://openapi.koreainvestment.com:9443"
        self.access_token = None
        self.token_file = "data/token.json"
        # print(self.app_key, self.app_secret, self.account_number)

    def load_token(self):
        """저장된 토큰 불러오기"""
        try:
            if os.path.exists(self.token_file):
                with open(self.token_file, "r") as f:
                    token_data = json.load(f)
                    if (
                        datetime.fromisoformat(token_data["expires_at"])
                        > datetime.now()
                    ):
                        self.access_token = token_data["access_token"]
                        print("\n=== 저장된 토큰 사용 ===")
                        print(f"Access Token: {self.access_token[:20]}...")
                        return self.access_token
        except Exception as e:
            print(f"토큰 로드 실패: {str(e)}")
        return None

    def save_token(self, token):
        """토큰 저장"""
        try:
            token_data = {
                "access_token": token,
                "expires_at": (datetime.now() + timedelta(hours=23)).isoformat(),
            }
            with open(self.token_file, "w") as f:
                json.dump(token_data, f)
        except Exception as e:
            print(f"토큰 저장 실패: {str(e)}")

    def get_access_token(self):
        """실제 토큰 발급"""
        # 저장된 토큰 확인
        token = self.load_token()
        if token:
            return token

        url = f"{self.base_url}/oauth2/tokenP"

        data = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }

        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            self.access_token = response.json()["access_token"]
            print("\n=== 토큰 발급 성공 ===")
            print(f"Access Token: {self.access_token[:20]}...")

            # 토큰 저장
            self.save_token(self.access_token)

            return self.access_token
        except Exception as e:
            print(f"\n=== 토큰 발급 실패 ===")
            print(f"Error: {str(e)}")
            if hasattr(e, "response"):
                print(f"Response: {e.response.text}")
            return None

    def get_account_info(self):
        """계좌 정보 조회"""
        if not self.access_token:
            self.access_token = self.get_access_token()
            if not self.access_token:
                return None

        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"

        headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {self.access_token}",
            "appKey": self.app_key,
            "appSecret": self.app_secret,
            "tr_id": "TTTC8434R",  # 실전투자용
        }

        params = {
            "CANO": self.account_number[:8],
            "ACNT_PRDT_CD": self.account_number[8:],
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "01",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }

        try:
            response = requests.get(url, headers=headers, params=params)
            print("\n=== API 응답 상태 코드 ===")
            print(f"Status Code: {response.status_code}")

            print("\n=== API 응답 헤더 ===")
            print(json.dumps(dict(response.headers), indent=2, ensure_ascii=False))

            print("\n=== API 응답 내용 ===")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))

            return response.json()
        except Exception as e:
            print(f"Error: {str(e)}")
            if hasattr(e, "response"):
                print(f"Response: {e.response.text}")
            return None


def main():
    print("=== 계좌 정보 조회 테스트 시작 ===")
    print(f"현재 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    test = AccountInfoTest()

    print("\n=== 환경 변수 확인 ===")
    print(f"APP_KEY: {test.app_key[:8]}...")
    print(f"APP_SECRET: {test.app_secret[:8]}...")
    print(f"ACCOUNT_NUMBER: {test.account_number}")

    print("\n=== 계좌 정보 조회 시도 ===")
    result = test.get_account_info()

    print("\n=== 테스트 완료 ===")


if __name__ == "__main__":
    main()
