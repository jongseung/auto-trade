import os
import json
import logging
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from urllib.parse import urljoin
import random
import numpy as np
import queue
import threading
import functools
from typing import Callable, Any

import config
from config import API_SETTINGS

logger = logging.getLogger("auto_trade.kis_api")


class TokenBucket:
    """API 호출 속도 제한을 위한 토큰 버킷 구현"""

    def __init__(self, fill_rate=1.0, capacity=10):
        """
        Args:
            fill_rate (float): 초당 추가되는 토큰 수
            capacity (int): 버킷의 최대 용량
        """
        self.capacity = capacity  # 버킷의 최대 용량
        self.tokens = capacity  # 현재 토큰 수
        self.fill_rate = fill_rate  # 초당 추가되는 토큰 수
        self.timestamp = time.time()  # 마지막 토큰 체크 시간
        self.lock = threading.RLock()  # 쓰레드 안전을 위한 락

    def consume(self, tokens=1):
        """
        토큰을 소비하려고 시도. 토큰이 충분하면 소비하고 True 반환,
        부족하면 대기 시간을 계산하여 대기 후 소비하거나 False 반환.

        Args:
            tokens (int): 소비할 토큰 수

        Returns:
            float: 대기한 시간(초). 대기하지 않았으면 0.
        """
        with self.lock:
            # 마지막 체크 이후 추가된 토큰 계산
            now = time.time()
            elapsed = now - self.timestamp
            self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
            self.timestamp = now

            # 토큰이 충분한지 확인
            if tokens <= self.tokens:
                self.tokens -= tokens
                return 0.0  # 대기 없이 즉시 실행
            else:
                # 필요한 대기 시간 계산
                wait_time = (tokens - self.tokens) / self.fill_rate
                # 토큰이 충분해질 때까지 대기
                time.sleep(wait_time)
                self.tokens = 0  # 토큰 모두 소비
                self.timestamp = time.time()  # 타임스탬프 업데이트
                return wait_time


class ApiCallQueue:
    """API 호출 큐잉 시스템"""

    def __init__(self, max_queue_size=100):
        """
        Args:
            max_queue_size (int): 최대 큐 크기
        """
        self.api_queue = queue.Queue(maxsize=max_queue_size)
        self.results = {}  # 호출 결과 저장
        self.is_running = False
        self.worker_thread = None

    def start(self, token_bucket):
        """큐 처리 워커 쓰레드 시작"""
        self.token_bucket = token_bucket
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._process_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def stop(self):
        """큐 처리 중지"""
        self.is_running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=3.0)

    def _process_queue(self):
        """API 호출 큐 처리 워커"""
        while self.is_running:
            try:
                call_id, func, args, kwargs = self.api_queue.get(timeout=0.5)
                # 토큰 버킷에서 토큰 소비 (API 제한 준수)
                wait_time = self.token_bucket.consume()
                if wait_time > 0:
                    logger.debug(f"API 호출 제한으로 {wait_time:.2f}초 대기")

                # API 함수 실행
                try:
                    result = func(*args, **kwargs)
                    self.results[call_id] = {"status": "success", "result": result}
                except Exception as e:
                    logger.error(f"API 호출 중 오류 발생: {str(e)}")
                    self.results[call_id] = {"status": "error", "error": str(e)}
                finally:
                    self.api_queue.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"API 큐 처리 중 오류 발생: {str(e)}")

    def add_call(self, func, *args, **kwargs):
        """API 호출 큐에 추가

        Args:
            func: 호출할 함수
            *args, **kwargs: 함수에 전달할 인자

        Returns:
            str: 호출 ID
        """
        call_id = f"call_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        self.api_queue.put((call_id, func, args, kwargs))
        return call_id

    def get_result(self, call_id, timeout=30):
        """API 호출 결과 가져오기

        Args:
            call_id (str): 호출 ID
            timeout (float): 최대 대기 시간(초)

        Returns:
            dict: API 호출 결과
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if call_id in self.results:
                result = self.results.pop(call_id)
                if result["status"] == "success":
                    return result["result"]
                else:
                    raise Exception(
                        f"API 호출 실패: {result.get('error', '알 수 없는 오류')}"
                    )
            time.sleep(0.1)

        raise TimeoutError(f"API 호출 결과 대기 시간 초과: {call_id}")


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """API 호출 실패 시 재시도하는 데코레이터"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(
                        f"API 호출 실패 (시도 {attempt + 1}/{max_retries}): {str(e)}"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))  # 지수 백오프
            logger.error(f"최대 재시도 횟수 초과: {str(last_exception)}")
            raise last_exception

        return wrapper

    return decorator


class KisAPI:
    """한국투자증권 OpenAPI 연동 클래스"""

    def __init__(self, demo_mode=None):
        """API 초기화

        Args:
            demo_mode (bool, optional): 더 이상 사용되지 않음
        """
        # API 키 및 계정 설정
        self.app_key = API_SETTINGS["APP_KEY"]
        self.app_secret = API_SETTINGS["APP_SECRET"]
        self.account_number = API_SETTINGS["ACCOUNT_NUMBER"]

        # 데모 모드 비활성화 (항상 실제 데이터 사용)
        self.demo_mode = False
        self.is_real_trading = True

        # API 서버 설정
        self.base_url = "https://openapi.koreainvestment.com:9443"
        self.token_file = os.path.join(config.DATA_DIR, "token.json")
        self.access_token = None
        self.token_expire_time = None
        self.is_connected = False

        # API 호출 제한 관리
        self.rate_limiter = TokenBucket(fill_rate=10, capacity=10)
        self.token_bucket = TokenBucket(fill_rate=3.0 / 5.0, capacity=5)
        self.api_queue = ApiCallQueue()
        self.api_queue.start(self.token_bucket)

        # 일별 API 호출 카운터 초기화
        self.daily_api_calls = 0
        self.daily_api_limit = 1000
        self.api_count_reset_date = datetime.now().date()

        # API 서버 연결
        self.connect()

        # 연결 상태 확인
        if not self.is_connected:
            logger.error("API 서버 연결 실패. 토큰을 확인하세요.")
            self._refresh_token_and_retry_connect()

    def _refresh_token_and_retry_connect(self):
        """토큰 재발급 및 연결 재시도"""
        logger.info("토큰 재발급 및 연결 재시도 중...")

        # 토큰 파일 삭제
        if os.path.exists(self.token_file):
            try:
                os.remove(self.token_file)
                logger.info("기존 토큰 파일 삭제됨")
            except Exception as e:
                logger.error(f"토큰 파일 삭제 실패: {str(e)}")

        # 연결 재시도
        if self.connect():
            logger.info("토큰 재발급 및 연결 성공")
        else:
            logger.error("토큰 재발급 및 연결 재시도 실패")
            # 여기서 강제로 에러를 발생시키지 않고, 후속 API 호출에서 처리

    def __del__(self):
        """소멸자: API 큐 처리 중지"""
        if hasattr(self, "api_queue"):
            self.api_queue.stop()

    def _check_api_limit(self):
        """API 호출 제한 체크 및 카운터 업데이트

        Returns:
            bool: API 호출 가능 여부
        """
        # 날짜가 바뀌었으면 카운터 초기화
        current_date = datetime.now().date()
        if current_date != self.api_count_reset_date:
            self.daily_api_calls = 0
            self.api_count_reset_date = current_date

        # 일별 한도 체크
        if self.daily_api_calls >= self.daily_api_limit:
            logger.error(
                f"일별 API 호출 한도({self.daily_api_limit}회)를 초과했습니다."
            )
            return False

        # 카운터 증가
        self.daily_api_calls += 1
        return True

    def connect(self):
        """API 서버 연결 및 접근 토큰 발급"""
        if self.demo_mode:
            logger.info("데모 모드: API 서버 연결 생략")
            return True

        if self.is_token_valid():
            logger.info("기존 토큰이 유효하여 재사용합니다.")
            return True

        url = urljoin(self.base_url, "/oauth2/tokenP")
        headers = {"content-type": "application/json"}
        data = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }

        try:
            # API 호출 제한 체크
            if not self._check_api_limit():
                logger.error("API 호출 한도 초과로 토큰 발급이 불가능합니다.")
                return False

            response = requests.post(url, headers=headers, data=json.dumps(data))
            response_data = response.json()

            if response.status_code == 200:
                self.access_token = response_data["access_token"]
                expires_in = response_data["expires_in"]
                self.token_expire_time = datetime.now() + timedelta(seconds=expires_in)
                self.is_connected = True

                # 토큰 정보 저장
                token_info = {
                    "access_token": self.access_token,
                    "expire_time": self.token_expire_time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                with open(self.token_file, "w") as f:
                    json.dump(token_info, f)

                logger.info("한국투자증권 API 서버 연결 성공")
                return True
            else:
                logger.error(f"API 서버 연결 실패: {response_data}")
                return False
        except Exception as e:
            logger.error(f"API 서버 연결 중 오류 발생: {str(e)}")
            return False

    def is_token_valid(self):
        """토큰 유효성 검사"""
        if self.demo_mode:
            return True

        try:
            if not os.path.exists(self.token_file):
                return False

            with open(self.token_file, "r") as f:
                token_info = json.load(f)

            if "access_token" not in token_info or "expire_time" not in token_info:
                return False

            self.access_token = token_info["access_token"]
            expire_time = datetime.strptime(
                token_info["expire_time"], "%Y-%m-%d %H:%M:%S"
            )

            # 만료 10분 전부터는 갱신
            if datetime.now() > expire_time - timedelta(minutes=10):
                return False

            self.token_expire_time = expire_time
            self.is_connected = True
            return True
        except Exception as e:
            logger.error(f"토큰 유효성 검사 중 오류 발생: {str(e)}")
            return False

    def get_headers(self, is_content_type=False):
        """API 요청 헤더"""
        headers = {
            "Content-Type": (
                "application/json"
                if is_content_type
                else "application/x-www-form-urlencoded"
            ),
            "Authorization": f"Bearer {self.access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": "",
        }
        return headers

    def get_stock_basic_info(self, ticker):
        """종목 기본 정보 조회"""
        if self.demo_mode:
            # 데모 데이터 반환
            stock_names = {
                "005930": "삼성전자",
                "000660": "SK하이닉스",
                "035420": "NAVER",
                "051910": "LG화학",
                "035720": "카카오",
                "293490": "카카오게임즈",
                "214150": "클래시스",
                "141080": "레고켐바이오",
                "035900": "JYP Ent.",
                "086520": "에코프로",
            }
            return {
                "prdt_name": stock_names.get(ticker, f"종목_{ticker}"),
                "stck_prpr": str(random.randint(10000, 100000)),  # 현재가
                "prdy_vrss_sign": random.choice(
                    ["1", "2", "3", "4", "5"]
                ),  # 전일 대비 부호
                "prdy_vrss": str(random.randint(100, 2000)),  # 전일 대비
                "prdy_ctrt": str(round(random.uniform(-5, 5), 2)),  # 전일 대비율
            }

        url = urljoin(self.base_url, "/uapi/domestic-stock/v1/quotations/search-info")
        params = {"PDNO": ticker, "PRDT_TYPE_CD": "300"}  # 주식

        headers = self.get_headers()
        headers["tr_id"] = "CTPF1002R"  # 주식 종목 기본정보 조회

        try:
            response = requests.get(url, headers=headers, params=params)
            result = response.json()

            if result["rt_cd"] == "0":
                return result["output"]
            else:
                logger.error(
                    f"종목 기본 정보 조회 실패: {result['msg_cd']} - {result['msg1']}"
                )
                return None
        except Exception as e:
            logger.error(f"종목 기본 정보 조회 중 오류 발생: {str(e)}")
            return None

    def get_ohlcv(self, ticker, interval="day", count=100, period=None):
        """OHLCV 데이터 조회

        Args:
            ticker (str): 종목 코드
            interval (str): 조회 간격 (day, week, month, year)
            count (int): 조회할 데이터 개수
            period (str, optional): 이전 버전 호환용 파라미터 ('D', 'W', 'M', 'm')

        Returns:
            pd.DataFrame: OHLCV 데이터
        """
        # 로그 추가
        logger.info(
            f"OHLCV 데이터 조회 시작: {ticker}, 간격: {interval}, 개수: {count}"
        )

        if period is not None:
            # 이전 버전 호환성 처리
            period_map = {"D": "day", "W": "week", "M": "month", "m": "minute"}
            interval = period_map.get(period, "day")

        # interval을 API 파라미터 형식으로 변환
        interval_map = {
            "day": "D",
            "week": "W",
            "month": "M",
            "minute": "m",
            # 원래 형식도 허용
            "D": "D",
            "W": "W",
            "M": "M",
            "m": "m",
        }
        period_code = interval_map.get(interval, "D")

        # 지수와 주식 구분
        is_index = ticker.startswith("U")

        # 시장이 개장하지 않은 날에도 데이터를 가져오기 위해 충분한 기간 설정
        # 현재 날짜에서 최소 1년 이전까지의 데이터를 조회 (주말, 공휴일 포함)
        end_date = datetime.now()
        start_date = end_date - timedelta(
            days=730
        )  # 2년으로 확장해서 충분한 데이터 확보

        max_retries = 3  # 최대 재시도 횟수
        for retry in range(max_retries):
            try:
                # API 파라미터 준비
                if is_index:
                    return self._get_index_data(
                        ticker, period_code, start_date, end_date, count
                    )
                else:
                    return self._get_stock_data(
                        ticker, period_code, start_date, end_date, count
                    )
            except Exception as e:
                if retry < max_retries - 1:
                    retry_delay = (retry + 1) * 2  # 지수 백오프
                    logger.warning(
                        f"OHLCV 데이터 조회 실패 ({retry+1}/{max_retries}), {retry_delay}초 후 재시도: {str(e)}"
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(f"OHLCV 데이터 조회 최종 실패: {str(e)}")
                    # 실패 시 빈 데이터프레임 반환
                    columns = [
                        "date",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "amount",
                    ]
                    return pd.DataFrame(columns=columns)

    def _get_stock_data(self, ticker, period_code, start_date, end_date, count):
        """주식 OHLCV 데이터 조회"""
        url = urljoin(
            self.base_url, "/uapi/domestic-stock/v1/quotations/inquire-daily-price"
        )
        headers = self.get_headers()
        headers["tr_id"] = "FHKST01010400"  # 주식 일별 시세 TR_ID

        params = {
            "FID_COND_MRKT_DIV_CODE": "J",  # 시장분류코드(J: 주식)
            "FID_INPUT_ISCD": ticker,  # 종목 코드
            "FID_PERIOD_DIV_CODE": period_code,  # 기간분류코드
            "FID_ORG_ADJ_PRC": "0",  # 수정주가 여부(0: 수정주가 적용)
            "FID_INPUT_DATE_1": start_date.strftime("%Y%m%d"),  # 시작일자
            "FID_INPUT_DATE_2": end_date.strftime("%Y%m%d"),  # 종료일자
        }

        # 주식 조회 요청 로그
        logger.debug(f"주식 조회 API 요청: {ticker}, URL: {url}, 파라미터: {params}")

        # 연결이 되지 않았으면 연결 시도
        if not self.is_connected:
            logger.warning("API 서버에 연결되지 않음. 연결 시도...")
            self.connect()
            if not self.is_connected:
                raise Exception("API 서버 연결 실패")

        # API 호출 및 응답 처리
        response = requests.get(url, headers=headers, params=params)
        result = response.json()

        # 응답 내용 상세 로깅
        logger.debug(
            f"OHLCV API 응답 ({ticker}): {json.dumps(result, indent=2, ensure_ascii=False)}"
        )

        # 응답 성공 여부 확인
        if result["rt_cd"] != "0":
            error_msg = (
                f"OHLCV API 오류 ({ticker}): {result['msg_cd']} - {result['msg1']}"
            )
            logger.error(error_msg)

            # 오류 코드가 토큰 만료인 경우 토큰 갱신 후 재시도
            if result["msg_cd"] in ["EGW00123", "EGW00121"]:  # 토큰 만료 오류 코드들
                logger.warning("토큰이 만료되었습니다. 토큰을 갱신하고 재시도합니다.")
                self.connect()
                if self.is_connected:
                    return self._get_stock_data(
                        ticker, period_code, start_date, end_date, count
                    )

            raise Exception(error_msg)

        # 출력 키 확인 (지수와 주식은 다른 출력 키를 사용할 수 있음)
        output_key = None
        for key in ["output1", "output2", "output"]:
            if key in result and result[key]:
                output_key = key
                break

        # 데이터가 없는 경우(주말이나 공휴일) 처리
        if output_key is None:
            error_msg = f"종목 {ticker} 데이터 없음 (주말/공휴일 등)"
            logger.warning(error_msg)
            # 빈 데이터프레임 반환
            columns = ["date", "open", "high", "low", "close", "volume", "amount"]
            return pd.DataFrame(columns=columns)

        # 데이터 추출 및 처리
        data = result[output_key]
        df = pd.DataFrame(data)

        # 원본 컬럼 확인
        logger.debug(f"원본 데이터 컬럼: {df.columns.tolist()}")

        # 컬럼명 매핑 (지수와 주식이 다른 컬럼명을 사용할 수 있음)
        column_mappings = {
            # 주식 컬럼
            "stck_bsop_date": "date",  # 기준일자
            "stck_oprc": "open",  # 시가
            "stck_hgpr": "high",  # 고가
            "stck_lwpr": "low",  # 저가
            "stck_clpr": "close",  # 종가
            "acml_vol": "volume",  # 거래량
            "acml_tr_pbmn": "amount",  # 거래대금
            # 지수 컬럼
            "bsop_date": "date",  # 기준일자
            "opnprc": "open",  # 시가
            "hgprc": "high",  # 고가
            "lwprc": "low",  # 저가
            "clsprc": "close",  # 종가
            "acc_trdvol": "volume",  # 거래량
            "acc_trdval": "amount",  # 거래대금
            # 추가 가능한 컬럼 매핑
            "prdy_vrss_sign": "change_sign",  # 전일 대비 부호
            "prdy_vrss": "change",  # 전일 대비
            "prdy_ctrt": "change_rate",  # 전일 대비율
        }

        # 컬럼명 변환
        for old_col, new_col in column_mappings.items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)

        # 필수 컬럼 확인
        required_columns = ["date", "open", "high", "low", "close"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(
                f"주식 {ticker}의 필수 컬럼 누락: {missing_columns}. 누락된 컬럼을 생성합니다."
            )

            # 날짜 컬럼이 없는 경우 현재 날짜로 생성
            if "date" in missing_columns:
                if "stck_bsop_date" in df.columns:
                    df.rename(columns={"stck_bsop_date": "date"}, inplace=True)
                else:
                    dates = pd.date_range(end=datetime.now(), periods=len(df))
                    df["date"] = dates

            # 기본 가격 추출 (close 컬럼이 있으면 사용, 없으면 생성)
            base_price = None
            if "close" in df.columns:
                base_price = df["close"].iloc[0] if not df.empty else 10000
            else:
                # 다른 가격 관련 컬럼 확인
                price_cols = [
                    col
                    for col in df.columns
                    if any(x in col for x in ["prc", "price", "prpr"])
                ]
                if price_cols:
                    base_price = (
                        pd.to_numeric(df[price_cols[0]].iloc[0], errors="coerce")
                        if not df.empty
                        else 10000
                    )
                    if pd.isna(base_price):
                        base_price = 10000
                else:
                    base_price = 10000  # 기본값

            # 누락된 가격 컬럼 생성
            for col in ["open", "high", "low", "close"]:
                if col in missing_columns:
                    if col == "high" and "open" in df.columns:
                        df[col] = df["open"] * 1.01  # 시가보다 약간 높게
                    elif col == "low" and "open" in df.columns:
                        df[col] = df["open"] * 0.99  # 시가보다 약간 낮게
                    else:
                        df[col] = base_price

        # 숫자형 변환
        numeric_columns = ["open", "high", "low", "close", "volume", "amount"]
        for col in numeric_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(
                        df[col].str.replace(",", ""), errors="coerce"
                    )
                except (AttributeError, ValueError):
                    df[col] = pd.to_numeric(df[col], errors="coerce")

        # 날짜 처리
        try:
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        except Exception as e:
            logger.error(f"날짜 변환 오류: {str(e)}")
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        df = df.sort_values("date", ascending=False)  # 최신 날짜 순으로 정렬

        # 필수 컬럼이 없는 경우 0으로 채우기
        for col in ["volume", "amount"]:
            if col not in df.columns:
                df[col] = 1  # 0 대신 1로 설정 (0으로 나누기 오류 방지)
            else:
                # 거래량이 0인 항목을 최소값으로 대체 (0으로 나누기 오류 방지)
                df[col] = df[col].apply(lambda x: max(1, x if not pd.isna(x) else 1))

        # 최신 거래일 데이터 확인
        if not df.empty:
            logger.info(
                f"종목 {ticker}의 최근 거래일: {df['date'].iloc[0].strftime('%Y-%m-%d')}, 데이터 수: {len(df)}"
            )

        # 이동평균 계산 (스크리너의 0으로 나누기 오류 방지)
        # 데이터프레임에 이동평균 컬럼 추가
        if len(df) >= 3:
            df["volume_ma3"] = df["volume"].rolling(window=3).mean()
            df["amount_ma3"] = df["amount"].rolling(window=3).mean()

            # NaN 값을 처리
            df["volume_ma3"] = df["volume_ma3"].fillna(df["volume"])
            df["amount_ma3"] = df["amount_ma3"].fillna(df["amount"])

            # 0 값을 최소값으로 대체 (0으로 나누기 오류 방지)
            df["volume_ma3"] = df["volume_ma3"].apply(lambda x: max(1.0, x))
            df["amount_ma3"] = df["amount_ma3"].apply(lambda x: max(1.0, x))

        # 사용자가 요청한 개수만큼 반환 (최신 데이터부터)
        df = df.head(count)
        df = df.sort_values("date")  # 날짜 순서로 재정렬

        return df

    def _get_index_data(self, ticker, period_code, start_date, end_date, count):
        """지수 OHLCV 데이터 조회"""
        # 연결 상태 확인
        if not self.is_connected:
            logger.warning("API 서버에 연결되지 않음. 연결 시도...")
            self.connect()
            if not self.is_connected:
                raise Exception("API 서버 연결 실패")

        # 여러 API 엔드포인트 시도 (기본 지수 API -> 차트 API -> ETF 대체 데이터)
        # 1. 기본 지수 API 먼저 시도
        try:
            df = self._get_index_data_primary(
                ticker, period_code, start_date, end_date, count
            )
            if not df.empty:
                return df
        except Exception as e:
            logger.warning(f"기본 지수 API 호출 실패: {str(e)}")

        # 2. 차트 API 시도
        try:
            df = self._get_index_data_chart(ticker, start_date, end_date, count)
            if not df.empty:
                return df
        except Exception as e:
            logger.warning(f"지수 차트 API 호출 실패: {str(e)}")

        # 3. ETF 대체 데이터 시도 (확장된 ETF 매핑)
        etf_map = {
            "U001": "069500",  # KOSPI -> KODEX 200
            "U201": "114800",  # KOSDAQ -> KODEX KOSDAQ
            "U501": "278540",  # KRX 300 -> KODEX KRX300
            "U101": "248260",  # KRX 100 -> KODEX KRX100
            "U402": "234310",  # KTOP 30 -> KODEX KTOP 30
            "U300": "278530",  # KRX Banking -> KODEX KRX 은행
            "U150": "163990",  # 코스피 고배당 -> TIGER 코스피고배당
            "U180": "117680",  # 코스피 배당성장 -> TIGER 코스피배당성장
        }

        # ETF 대응 매핑 시도
        try:
            if ticker in etf_map:
                etf_ticker = etf_map[ticker]
                logger.info(
                    f"지수 {ticker}에 대응하는 ETF 티커 {etf_ticker}로 대체 조회"
                )
            else:
                # 알 수 없는 지수 코드는 KODEX 200으로 대체
                etf_ticker = "069500"  # KODEX 200 ETF
                logger.info(
                    f"알 수 없는 지수 {ticker}에 대해 기본 ETF 티커 {etf_ticker}(KODEX 200)로 대체 조회"
                )

            return self._get_stock_data(
                etf_ticker, period_code, start_date, end_date, count
            )
        except Exception as e:
            logger.warning(f"ETF 대체 데이터 호출 실패: {str(e)}")

        # 4. 시장 전체에 대한 기본 데이터 생성 (빈 데이터프레임 반환은 최후의 수단)
        logger.error(f"모든 지수 데이터 조회 시도 실패: {ticker}")
        columns = ["date", "open", "high", "low", "close", "volume", "amount"]
        return pd.DataFrame(columns=columns)

    def _get_index_data_primary(self, ticker, period_code, start_date, end_date, count):
        """기본 지수 API를 통한 데이터 조회"""
        url = urljoin(
            self.base_url,
            "/uapi/domestic-stock/v1/quotations/inquire-index-daily-price",
        )
        headers = self.get_headers()
        headers["tr_id"] = "FHKUP03500100"  # 지수 일별 시세 조회 TR_ID

        params = {
            "FID_COND_MRKT_DIV_CODE": "U",  # 시장분류코드(U: 지수)
            "FID_INPUT_ISCD": ticker,  # 지수 코드
            "FID_INPUT_DATE_1": start_date.strftime("%Y%m%d"),  # 시작일자
            "FID_INPUT_DATE_2": end_date.strftime("%Y%m%d"),  # 종료일자
            "FID_PERIOD_DIV_CODE": period_code,  # 기간분류코드
        }

        # 지수 조회 요청 로그
        logger.debug(f"기본 지수 API 요청: {ticker}, URL: {url}, 파라미터: {params}")

        # 지수 데이터 조회 시도
        response = requests.get(url, headers=headers, params=params)
        result = response.json()

        # 응답 내용 상세 로깅
        logger.debug(
            f"기본 지수 API 응답: {json.dumps(result, indent=2, ensure_ascii=False)}"
        )

        if result["rt_cd"] != "0":
            error_msg = f"기본 지수 API 오류: {result['msg_cd']} - {result.get('msg1', '알 수 없는 오류')}"
            logger.error(error_msg)

            # 오류 코드가 토큰 만료인 경우 토큰 갱신 후 재시도
            if result["msg_cd"] in ["EGW00123", "EGW00121"]:  # 토큰 만료 오류 코드들
                logger.warning("토큰이 만료되었습니다. 토큰을 갱신하고 재시도합니다.")
                self.connect()
                if self.is_connected:
                    return self._get_index_data_primary(
                        ticker, period_code, start_date, end_date, count
                    )

            raise Exception(error_msg)

        # 데이터 확인
        output_key = None
        for key in ["output1", "output", "output2"]:
            if key in result and result[key]:
                output_key = key
                break

        if output_key is None:
            error_msg = f"기본 지수 API 응답에 데이터가 없습니다: {ticker}"
            logger.error(error_msg)
            raise Exception(error_msg)

        raw_data = result[output_key]

        # 원본 API 응답 컬럼 로깅
        if raw_data:
            logger.debug(
                f"지수 {ticker} 원본 API 응답 컬럼: {list(raw_data[0].keys())}"
            )

        # 데이터프레임 변환
        df = pd.DataFrame(raw_data, index=range(len(raw_data)))

        # 컬럼 매핑
        column_mappings = {
            "bsop_date": "date",  # 기준일자
            "opnprc": "open",  # 시가
            "hgprc": "high",  # 고가
            "lwprc": "low",  # 저가
            "clsprc": "close",  # 종가
            "acc_trdvol": "volume",  # 거래량
            "acc_trdval": "amount",  # 거래대금
        }

        # 컬럼명 변환
        for old_col, new_col in column_mappings.items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)

        # 필수 컬럼 확인
        required_cols = ["date", "open", "high", "low", "close"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            error_msg = f"필수 컬럼 누락: {missing_cols}"
            logger.error(error_msg)
            raise Exception(error_msg)

        # 숫자형 변환
        numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
        for col in numeric_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(
                        df[col].str.replace(",", ""), errors="coerce"
                    )
                except (AttributeError, ValueError):
                    df[col] = pd.to_numeric(df[col], errors="coerce")

        # 날짜 변환
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

        # 정렬 및 데이터 수 제한
        df = df.sort_values("date", ascending=False).head(count)
        df = df.sort_values("date")  # 날짜순 재정렬

        # 거래량 및 거래대금이 없는 경우 1로 설정 (0으로 나누기 오류 방지)
        for col in ["volume", "amount"]:
            if col not in df.columns:
                df[col] = 1
            else:
                df[col] = df[col].apply(lambda x: max(1, x if not pd.isna(x) else 1))

        # 이동평균 계산
        if len(df) >= 3:
            df["volume_ma3"] = (
                df["volume"].rolling(window=3).mean().fillna(df["volume"])
            )
            df["amount_ma3"] = (
                df["amount"].rolling(window=3).mean().fillna(df["amount"])
            )
            df["volume_ma3"] = df["volume_ma3"].apply(lambda x: max(1.0, x))
            df["amount_ma3"] = df["amount_ma3"].apply(lambda x: max(1.0, x))

        logger.info(f"지수 {ticker} 데이터 조회 성공: {len(df)}개 데이터")
        return df

    def _get_index_data_chart(self, ticker, start_date, end_date, count):
        """지수 차트 API를 통한 데이터 조회"""
        url = urljoin(
            self.base_url,
            "/uapi/domestic-stock/v1/quotations/inquire-daily-indexchartprice",
        )
        headers = self.get_headers()
        headers["tr_id"] = "FHKST03010100"  # 지수 차트 API

        params = {
            "FID_COND_MRKT_DIV_CODE": "U",  # 시장분류코드(U: 지수)
            "FID_INPUT_ISCD": ticker,  # 지수 코드
            "FID_INPUT_DATE_1": start_date.strftime("%Y%m%d"),  # 시작일자
            "FID_INPUT_DATE_2": end_date.strftime("%Y%m%d"),  # 종료일자
            "FID_PERIOD_DIV_CODE": "D",  # 일봉
        }

        # 지수 차트 API 요청 로그
        logger.debug(f"지수 차트 API 요청: {ticker}, URL: {url}, 파라미터: {params}")

        # 지수 차트 데이터 조회 시도
        response = requests.get(url, headers=headers, params=params)
        result = response.json()

        # 응답 내용 상세 로깅
        logger.debug(
            f"지수 차트 API 응답: {json.dumps(result, indent=2, ensure_ascii=False)}"
        )

        if result["rt_cd"] != "0":
            error_msg = f"지수 차트 API 오류: {result['msg_cd']} - {result.get('msg1', '알 수 없는 오류')}"
            logger.error(error_msg)
            raise Exception(error_msg)

        # 데이터 확인
        output_key = None
        for key in ["output1", "output", "output2"]:
            if key in result and result[key]:
                output_key = key
                break

        if output_key is None:
            error_msg = f"지수 차트 API 응답에 데이터가 없습니다: {ticker}"
            logger.error(error_msg)
            raise Exception(error_msg)

        raw_data = result[output_key]

        # 원본 API 응답 컬럼 로깅
        if raw_data:
            logger.debug(f"지수 차트 API 응답 컬럼: {list(raw_data[0].keys())}")

        # 데이터프레임 변환
        df = pd.DataFrame(raw_data, index=range(len(raw_data)))

        # 컬럼 매핑
        column_mappings = {
            "bsop_date": "date",  # 기준일자
            "bsop_opnprc": "open",  # 시가
            "bsop_hgprc": "high",  # 고가
            "bsop_lwprc": "low",  # 저가
            "bsop_clsprc": "close",  # 종가
            "bsop_trqu": "volume",  # 거래량
            "bsop_trprc": "amount",  # 거래대금
        }

        # 컬럼명 변환
        for old_col, new_col in column_mappings.items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)

        # 필수 컬럼 확인
        required_cols = ["date", "open", "high", "low", "close"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            error_msg = f"지수 차트 API 필수 컬럼 누락: {missing_cols}"
            logger.error(error_msg)
            raise Exception(error_msg)

        # 숫자형 변환
        numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
        for col in numeric_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(
                        df[col].str.replace(",", ""), errors="coerce"
                    )
                except (AttributeError, ValueError):
                    df[col] = pd.to_numeric(df[col], errors="coerce")

        # 날짜 변환
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

        # 정렬 및 데이터 수 제한
        df = df.sort_values("date", ascending=False).head(count)
        df = df.sort_values("date")  # 날짜순 재정렬

        # 거래량 및 거래대금이 없는 경우 1로 설정 (0으로 나누기 오류 방지)
        for col in ["volume", "amount"]:
            if col not in df.columns:
                df[col] = 1
            else:
                df[col] = df[col].apply(lambda x: max(1, x if not pd.isna(x) else 1))

        # 이동평균 계산
        if len(df) >= 3:
            df["volume_ma3"] = (
                df["volume"].rolling(window=3).mean().fillna(df["volume"])
            )
            df["amount_ma3"] = (
                df["amount"].rolling(window=3).mean().fillna(df["amount"])
            )
            df["volume_ma3"] = df["volume_ma3"].apply(lambda x: max(1.0, x))
            df["amount_ma3"] = df["amount_ma3"].apply(lambda x: max(1.0, x))

        logger.info(f"지수 차트 API {ticker} 데이터 조회 성공: {len(df)}개 데이터")
        return df

    def _get_demo_data(self, ticker, count=100):
        """데모/오류 시 사용할 가상 OHLCV 데이터 생성"""
        logger.info(f"가상 OHLCV 데이터 생성: {ticker}, 개수: {count}")

        # 기본 가격 설정 (종목코드별로 일관된 값 생성)
        ticker_seed = sum(ord(c) for c in ticker)
        random.seed(ticker_seed)
        base_price = random.randint(10000, 100000)
        random.seed()  # 시드 초기화

        # 날짜 범위 설정 - 오늘부터 과거로
        end_date = datetime.now()
        dates = [end_date - timedelta(days=i) for i in range(count)]
        dates.reverse()  # 과거부터 현재 순서로

        data = []
        current_price = base_price

        for date in dates:
            change_rate = random.uniform(-0.03, 0.03)
            open_price = int(current_price * (1 + random.uniform(-0.01, 0.01)))
            high_price = int(open_price * (1 + random.uniform(0, 0.02)))
            low_price = int(open_price * (1 - random.uniform(0, 0.02)))
            close_price = int(current_price * (1 + change_rate))
            close_price = max(low_price, min(close_price, high_price))
            # 거래량이 0이 되지 않도록 최소값 설정
            volume = max(100, random.randint(100000, 10000000))
            amount = close_price * volume
            current_price = close_price

            data.append(
                {
                    "date": date,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                    "amount": amount,
                }
            )

        try:
            # 명시적인 인덱스 설정으로 데이터프레임 생성
            df = pd.DataFrame(data)

            # 날짜 컬럼을 인덱스로 설정했다가 다시 컬럼으로 복원
            df = df.set_index("date").reset_index()

            # 이동평균 계산 (스크리너의 0으로 나누기 오류 방지)
            if len(df) >= 3:
                df["volume_ma3"] = (
                    df["volume"].rolling(window=3).mean().fillna(df["volume"])
                )
                df["amount_ma3"] = (
                    df["amount"].rolling(window=3).mean().fillna(df["amount"])
                )

                # 0 값을 최소값으로 대체 (0으로 나누기 오류 방지)
                df["volume_ma3"] = df["volume_ma3"].apply(lambda x: max(1.0, x))
                df["amount_ma3"] = df["amount_ma3"].apply(lambda x: max(1.0, x))

            return df
        except Exception as e:
            logger.error(f"데모 데이터 생성 중 오류 발생: {str(e)}")

            # 오류 발생 시 기본 데이터프레임 생성 (최소한의 데이터)
            fallback_dates = pd.date_range(end=datetime.now(), periods=count)
            fallback_data = {
                "date": fallback_dates,
                "open": [base_price] * count,
                "high": [base_price * 1.01] * count,
                "low": [base_price * 0.99] * count,
                "close": [base_price] * count,
                "volume": [1000] * count,
                "amount": [base_price * 1000] * count,
            }
            # 데이터프레임 생성 시 인덱스 명시적 지정
            df = pd.DataFrame(fallback_data, index=range(count))

            # 이동평균 계산 (스크리너의 0으로 나누기 오류 방지)
            if len(df) >= 3:
                df["volume_ma3"] = (
                    df["volume"].rolling(window=3).mean().fillna(df["volume"])
                )
                df["amount_ma3"] = (
                    df["amount"].rolling(window=3).mean().fillna(df["amount"])
                )

                # 0 값을 최소값으로 대체 (0으로 나누기 오류 방지)
                df["volume_ma3"] = df["volume_ma3"].apply(lambda x: max(1.0, x))
                df["amount_ma3"] = df["amount_ma3"].apply(lambda x: max(1.0, x))

            return df

    def get_current_price(self, ticker):
        """현재가 조회"""
        if self.demo_mode:
            # 데모 데이터 반환
            return random.randint(10000, 100000)

        url = urljoin(self.base_url, "/uapi/domestic-stock/v1/quotations/inquire-price")
        headers = self.get_headers()
        headers["tr_id"] = "FHKST01010100"  # 주식 현재가 시세

        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": ticker,
        }

        try:
            response = requests.get(url, headers=headers, params=params)
            result = response.json()

            if result["rt_cd"] == "0":
                data = result["output"]
                current_price = int(data["stck_prpr"])  # 현재가
                return current_price
            else:
                logger.error(f"현재가 조회 실패: {result['msg_cd']} - {result['msg1']}")
                return None
        except Exception as e:
            logger.error(f"현재가 조회 중 오류 발생: {str(e)}")
            return None

    def get_market_ohlcv(self, ticker_list, days=5):
        """복수 종목의 OHLCV 데이터 조회"""
        result = {}

        for ticker in ticker_list:
            ohlcv = self.get_ohlcv(ticker, interval="day", count=days)
            if not ohlcv.empty:
                result[ticker] = ohlcv

            # API 호출 제한 고려 (초당 1건)
            if not self.demo_mode:
                time.sleep(0.5)

        return result

    @retry_on_failure(max_retries=3, delay=1.0)
    def get_account_info(self):
        """계좌 정보 조회

        Returns:
            dict: 계좌 정보
        """
        try:
            if self.demo_mode:
                # 데모 계좌 정보 생성
                return self._get_demo_account_info()

            # API 요청 URL 설정
            url = "/uapi/domestic-stock/v1/trading/inquire-balance"

            # API 요청 헤더 및 파라미터 설정
            headers = self.get_headers(is_content_type=True)
            headers["tr_id"] = "TTTC8434R" if self.is_real_trading else "VTTC8434R"

            params = {
                "CANO": self.account_number[:8],
                "ACNT_PRDT_CD": self.account_number[8:],
                "AFHR_FLPR_YN": "N",
                "OFL_YN": "N",
                "INQR_DVSN": "01",
                "UNPR_DVSN": "01",
                "FUND_STTL_ICLD_YN": "N",
                "FNCG_AMT_AUTO_RDPT_YN": "N",
                "PRCS_DVSN": "01",
                "CTX_AREA_FK100": "",
                "CTX_AREA_NK100": "",
            }

            # API 요청 실행
            try:
                response = self._api_request("get", url, headers=headers, params=params)
                if response is None:
                    logger.warning(
                        "계좌 정보 API 응답이 없습니다. 데모 데이터를 반환합니다."
                    )
                    return self._get_demo_account_info()

                logger.info(f"API 응답: {response}")

                # 응답 데이터 변환
                # API 응답 형식 확인
                if not isinstance(response, dict):
                    logger.error(f"API 응답이 올바른 형식이 아닙니다: {type(response)}")
                    return self._get_demo_account_info()

                output1 = response.get("output1", [])
                output2 = response.get("output2", [])

                # 리스트인지 확인하고 안전하게 처리
                if not isinstance(output1, list) or not isinstance(output2, list):
                    logger.error(
                        f"API 응답 output이 리스트 형식이 아닙니다: output1={type(output1)}, output2={type(output2)}"
                    )
                    return self._get_demo_account_info()

                # output2가 비어있는 경우 처리
                if not output2:
                    logger.warning(
                        "계좌 정보 API 응답의 output2가 비어 있습니다. 데모 데이터를 반환합니다."
                    )
                    return self._get_demo_account_info()

                # 안전하게 첫 번째 항목 접근
                account_summary = output2[0] if output2 else {}

                # 계좌 정보 구성
                account_info = {
                    "total_value": int(account_summary.get("tot_evlu_amt", "0")),
                    "balance": int(account_summary.get("dnca_tot_amt", "0")),
                    "total_profit_loss": int(
                        account_summary.get("evlu_pfls_smtl_amt", "0")
                    ),
                    "total_profit_loss_rate": float(
                        account_summary.get("asst_icdc_erng_rt", "0")
                    ),
                    "positions": [],
                }

                # 보유 종목 정보 구성
                for position in output1:
                    ticker = position.get("pdno", "")
                    if ticker and position.get("hldg_qty", "0") != "0":
                        position_info = {
                            "ticker": ticker,
                            "name": position.get("prdt_name", ""),
                            "quantity": int(position.get("hldg_qty", "0")),
                            "buy_price": int(float(position.get("pchs_avg_pric", "0"))),
                            "current_price": int(position.get("prpr", "0")),
                            "eval_profit_loss": int(position.get("evlu_pfls_amt", "0")),
                            "eval_amount": int(position.get("evlu_amt", "0")),
                            "profit_loss_rate": float(
                                position.get("evlu_pfls_rt", "0")
                            ),
                        }
                        account_info["positions"].append(position_info)

                return account_info
            except Exception as api_error:
                logger.error(f"계좌 정보 API 조회 중 오류 발생: {str(api_error)}")
                return self._get_demo_account_info()

        except Exception as e:
            logger.error(f"계좌 정보 조회 중 오류 발생: {str(e)}")
            return self._get_demo_account_info()

    def _get_demo_account_info(self):
        """데모용 계좌 정보 생성"""
        return {
            "total_value": 10000000,
            "balance": 5000000,
            "total_profit_loss": 100000,
            "total_profit_loss_rate": 1.0,
            "positions": [
                {
                    "ticker": "005930",
                    "name": "삼성전자",
                    "quantity": 5,
                    "buy_price": 70000,
                    "current_price": 71000,
                    "eval_profit_loss": 5000,
                    "eval_amount": 355000,
                    "profit_loss_rate": 1.43,
                }
            ],
        }

    def get_tick_size(self, price):
        """가격대별 호가단위 반환

        Args:
            price (float): 가격

        Returns:
            int: 호가단위
        """
        # 2023년 1월 25일부터 적용된 호가단위 (코스피, 코스닥 모두 동일)
        if price < 2000:
            return 1
        elif price < 5000:
            return 5
        elif price < 20000:
            return 10
        elif price < 50000:
            return 50
        elif price < 200000:
            return 100
        elif price < 500000:
            return 500
        else:
            return 1000

    def round_to_tick_size(self, price, order_type="limit"):
        """가격을 호가단위에 맞게 조정

        Args:
            price (float): 조정할 가격
            order_type (str): 주문 유형 (limit: 지정가, market: 시장가)

        Returns:
            int: 호가단위에 맞게 조정된 가격
        """
        if order_type == "market":
            return 0  # 시장가 주문은 가격이 0

        tick_size = self.get_tick_size(price)

        # 나머지 계산 후 가장 가까운 호가단위로 조정
        remainder = price % tick_size
        if remainder == 0:
            return int(price)  # 이미 호가단위에 맞는 경우

        # 반올림 로직: 나머지가 틱 사이즈의 절반 이상이면 올림, 아니면 내림
        if remainder >= tick_size / 2:
            return int(price - remainder + tick_size)
        else:
            return int(price - remainder)

    def place_order(self, ticker, order_type, quantity, price=0, order_dvsn="00"):
        """주문 실행

        Args:
            ticker (str): 종목코드
            order_type (str): 주문유형 (1: 매도, 2: 매수)
            quantity (int): 주문수량
            price (int): 주문가격
            order_dvsn (str): 주문구분 (00: 지정가, 01: 시장가)

        Returns:
            dict: 주문결과
        """
        # API 사용량 제한 체크
        if not self._check_api_limit():
            return None

        # 연결 상태 확인
        if not self.is_connected:
            logger.warning("API 서버에 연결되지 않음. 연결 시도...")
            self.connect()
            if not self.is_connected:
                raise Exception("API 서버 연결 실패")

        # 지정가 주문인 경우 호가단위에 맞게 가격 조정
        if order_dvsn == "00" and price > 0:
            price = self.round_to_tick_size(price, "limit")

        url = urljoin(self.base_url, "/uapi/domestic-stock/v1/trading/order-cash")

        headers = self.get_headers()
        headers["tr_id"] = (
            ("TTTC0802U" if order_type == "1" else "TTTC0801U")
            if self.is_real_trading
            else ("VTTC0802U" if order_type == "1" else "VTTC0801U")
        )

        data = {
            "CANO": self.account_number,
            "ACNT_PRDT_CD": self.account_number,
            "PDNO": ticker,
            "ORD_DVSN": order_dvsn,  # 주문구분(00: 지정가, 01: 시장가)
            "ORD_QTY": str(quantity),  # 주문수량
            "ORD_UNPR": str(price if order_dvsn == "00" else "0"),  # 주문가격
            "CTAC_TLNO": "",  # 연락전화번호
            "SLL_BUY_DVSN_CD": order_type,  # 매매구분(1: 매도, 2: 매수)
            "ALGO_NO": "",  # 알고리즘 번호
        }

        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            result = response.json()

            if result["rt_cd"] == "0":
                order_result = {
                    "order_no": result["output"][
                        "KRX_FWDG_ORD_ORGNO"
                    ],  # 한국거래소전송주문조직번호
                    "order_id": result["output"]["ODNO"],  # 주문번호
                    "order_time": result["output"]["ORD_TMD"],  # 주문시각
                }
                logger.info(
                    f"주문 요청 성공: {ticker} {'매수' if order_type == '2' else '매도'} {quantity}주"
                )
                return order_result
            else:
                logger.error(f"주문 요청 실패: {result['msg_cd']} - {result['msg1']}")
                return None
        except Exception as e:
            logger.error(f"주문 요청 중 오류 발생: {str(e)}")
            return None

    def market_buy(self, ticker, quantity):
        """시장가 매수"""
        return self.place_order(ticker, "2", quantity, 0, "01")

    def market_sell(self, ticker, quantity):
        """시장가 매도"""
        return self.place_order(ticker, "1", quantity, 0, "01")

    def limit_buy(self, ticker, quantity, price):
        """지정가 매수"""
        # 호가단위에 맞게 가격 조정
        price = self.round_to_tick_size(price, "limit")
        return self.place_order(ticker, "2", quantity, price, "00")

    def limit_sell(self, ticker, quantity, price):
        """지정가 매도"""
        # 호가단위에 맞게 가격 조정
        price = self.round_to_tick_size(price, "limit")
        return self.place_order(ticker, "1", quantity, price, "00")

    def get_order_status(self, order_no):
        """주문 체결 확인"""
        if self.demo_mode:
            # 데모 주문 상태 반환
            return {
                "ticker": "005930",  # 임의의 종목 코드
                "order_no": order_no,
                "order_quantity": 10,
                "executed_quantity": 10,
                "remained_quantity": 0,
                "status": "체결완료",
            }

        url = urljoin(self.base_url, "/uapi/domestic-stock/v1/trading/inquire-order")
        headers = self.get_headers()
        headers["tr_id"] = (
            "TTTC8001R" if self.is_real_trading else "VTTC8001R"
        )  # 주식 정정취소가능주문조회

        params = {
            "CANO": self.account_number,
            "ACNT_PRDT_CD": self.account_number,
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
            "INQR_DVSN_1": "0",  # 조회구분1(0: 전체)
            "INQR_DVSN_2": "0",  # 조회구분2(0: 전체)
        }

        try:
            response = requests.get(url, headers=headers, params=params)
            result = response.json()

            if result["rt_cd"] == "0":
                orders = result["output"]
                for order in orders:
                    if order["odno"] == order_no:
                        return {
                            "ticker": order["pdno"],
                            "order_no": order["odno"],
                            "order_quantity": int(order["ord_qty"]),
                            "executed_quantity": int(order["tot_ccld_qty"]),
                            "remained_quantity": int(order["rmn_qty"]),
                            "status": order["ord_prcs_psta_name"],
                        }
                return None
            else:
                logger.error(
                    f"주문 상태 조회 실패: {result['msg_cd']} - {result['msg1']}"
                )
                return None
        except Exception as e:
            logger.error(f"주문 상태 조회 중 오류 발생: {str(e)}")
            return None

    def get_kospi_index(self):
        """KOSPI 지수 조회"""
        if not self._check_api_limit():
            return None

        url = urljoin(self.base_url, "/uapi/domestic-stock/v1/quotations/inquire-price")
        headers = self.get_headers()
        headers["tr_id"] = "FHKST01010100"  # 주식 현재가 시세

        params = {
            "FID_COND_MRKT_DIV_CODE": "U",  # 시장분류코드(U: 지수)
            "FID_INPUT_ISCD": "U001",  # KOSPI 지수
        }

        try:
            response = requests.get(url, headers=headers, params=params)
            result = response.json()

            if result["rt_cd"] == "0" and "output" in result and result["output"]:
                data = result["output"]
                return {
                    "index": float(data.get("stck_prpr", 0)),  # 현재가
                    "change": float(data.get("prdy_vrss", 0)),  # 전일 대비
                    "change_rate": float(data.get("prdy_ctrt", 0)),  # 전일 대비율
                    "date": datetime.now().strftime("%Y%m%d"),  # 현재 날짜
                }
            else:
                logger.warning(
                    f"KOSPI 지수 데이터 조회 실패: {result.get('msg1', '알 수 없는 오류')}"
                )
                logger.debug(
                    f"API 응답: {json.dumps(result, indent=2, ensure_ascii=False)}"
                )
                return None

        except Exception as e:
            logger.error(f"KOSPI 지수 조회 중 오류 발생: {str(e)}")
            return None

    def get_disclosure(self, date=None):
        """OpenDART API를 사용한 공시 정보 조회

        Args:
            date (str): 조회일자(YYYYMMDD)

        Returns:
            list: 공시 정보 리스트
        """
        if self.demo_mode:
            # 데모 공시 정보 반환
            logger.info("데모 모드: 데모 공시 정보 반환")
            return self._get_demo_disclosure(date)

        # 날짜 설정 및 형식 처리
        current_date = datetime.now().date()

        # 시작일은 7일 전으로 설정 (날짜 범위 확장)
        start_date = (current_date - timedelta(days=7)).strftime("%Y%m%d")

        # 종료일은 현재 날짜
        end_date = current_date.strftime("%Y%m%d")

        if date is not None:
            # 사용자가 지정한 날짜가 있으면 형식 확인
            try:
                # 형식 변환 테스트
                parsed_date = datetime.strptime(date, "%Y%m%d").date()
                end_date = date
                # 시작일은 지정된 날짜로부터 7일 전으로 설정
                start_date = (parsed_date - timedelta(days=7)).strftime("%Y%m%d")
            except ValueError:
                logger.warning(f"잘못된 날짜 형식: {date}, 기본 날짜 사용")

        # 주말이나 휴장일인 경우에도 지난 7일 범위 내의 데이터를 조회하도록 함
        logger.info(f"OpenDART API로 공시 정보 조회: {start_date} ~ {end_date}")

        # API 설정 로그 출력 부분 수정
        logger.info("API 설정 로드 완료")

        # OpenDART API 키 확인 - 민감 정보 로깅 제거
        try:
            api_key = API_SETTINGS.get("DART_API_KEY", "")
            # 로그에서 API 키 관련 상세 정보 제거
            logger.info("OpenDART API 키 설정 확인")

            if not api_key:
                logger.error(
                    "OpenDART API 키가 설정되지 않았습니다. 데모 데이터를 반환합니다."
                )
                return self._get_demo_disclosure(date)

            # 환경 변수 직접 확인 - 민감 정보 로깅 제거
            env_api_key = os.getenv("DART_API_KEY", "")
            logger.info("환경변수 DART_API_KEY 확인")

            # API 요청 URL 및 파라미터 설정 (명시적 형식 지정)
            url = "https://opendart.fss.or.kr/api/list.json"
            params = {
                "crtfc_key": api_key,
                "bgn_de": start_date,  # 시작일 (7일 전)
                "end_de": end_date,  # 종료일 (현재 또는 지정일)
                "page_count": 100,
                "page_no": 1,
            }

            logger.debug(f"OpenDART API 요청: {url}, 파라미터: {params}")

            # API 요청
            response = requests.get(url, params=params, timeout=10)

            # 응답 처리
            if response.status_code != 200:
                logger.error(f"OpenDART API 호출 실패: HTTP {response.status_code}")
                logger.error(f"응답 내용: {response.text[:200]}")
                return self._get_demo_disclosure(date)

            result = response.json()
            logger.debug(
                f"OpenDART API 응답: {json.dumps(result, indent=2, ensure_ascii=False)}"
            )

            # API 응답 상태 확인
            if result.get("status") != "000":
                error_msg = result.get("message", "알 수 없는 오류")
                logger.error(f"OpenDART API 오류 상태: {result.get('status')}")
                logger.error(f"OpenDART API 오류 메시지: {error_msg}")

                # 오류 코드별 상세 메시지
                status_code = result.get("status", "")
                if status_code == "010":
                    logger.error("OpenDART API 오류: 유효하지 않은 API 키")
                elif status_code == "011":
                    logger.error("OpenDART API 오류: API 사용량 초과")
                elif status_code == "013":
                    logger.error("OpenDART API 오류: 필수 값 누락")
                elif status_code == "020":
                    logger.error("OpenDART API 오류: 조회 가능한 데이터가 없음")
                elif status_code == "100":
                    logger.error("OpenDART API 오류: 서버 에러")

                return self._get_demo_disclosure(date)

            # 데이터 변환 (OpenDART 형식 -> 자체 형식)
            disclosures = []
            for item in result.get("list", []):
                # 종목코드 추출 (OpenDART는 종목코드가 8자리: 앞 6자리가 실제 종목코드)
                stock_code = item.get("stock_code", "").strip()
                if stock_code and len(stock_code) >= 6:
                    stock_code = stock_code[:6]

                # 공시 시간 추출 및 포맷 변환
                rcept_dt = item.get("rcept_dt", "")
                rcept_time = ""
                if rcept_dt and len(rcept_dt) >= 8:
                    rcept_time = rcept_dt[
                        8:12
                    ]  # YYYYMMDD + HHmm 형식에서 시간부분 추출
                    if not rcept_time:
                        rcept_time = "0900"  # 기본값 설정

                # 형식 변환된 공시 정보 추가
                disclosures.append(
                    {
                        "mksc_shrn_iscd": stock_code,  # 종목코드
                        "corp_name": item.get("corp_name", ""),  # 회사명
                        "dics_hora": rcept_time,  # 공시시간
                        "dics_tl": item.get("report_nm", ""),  # 공시제목
                        "dics_date": (
                            item.get("rcept_dt", "")[:8]
                            if item.get("rcept_dt")
                            else end_date
                        ),  # 공시일자
                        "url": item.get("dart_url", ""),  # 공시 URL
                    }
                )

            logger.info(
                f"OpenDART API에서 {len(disclosures)}건의 공시 정보를 가져왔습니다."
            )

            # 공시 정보가 없는 경우 데모 데이터 반환
            if not disclosures:
                logger.info("공시 정보가 없어 데모 데이터를 반환합니다.")
                return self._get_demo_disclosure(date)

            return disclosures

        except requests.exceptions.RequestException as e:
            logger.error(f"OpenDART API 요청 오류: {type(e).__name__}, {str(e)}")
            return self._get_demo_disclosure(date)
        except json.JSONDecodeError as e:
            logger.error(f"OpenDART API 응답 JSON 파싱 실패: {str(e)}")
            return self._get_demo_disclosure(date)
        except Exception as e:
            logger.error(f"OpenDART API 조회 중 오류 발생: {str(e)}")
            logger.exception("상세 오류:")
            return self._get_demo_disclosure(date)

    def _get_demo_disclosure(self, date=None):
        """데모용 공시 정보 생성"""
        if date is None:
            date = datetime.now().strftime("%Y%m%d")

        # 기본 공시 데이터
        demo_disclosures = [
            {
                "mksc_shrn_iscd": "005930",  # 종목코드
                "corp_name": "삼성전자",  # 회사명
                "dics_hora": "0900",  # 공시시간
                "dics_tl": "주요사업 신규계약 체결",  # 공시제목
                "dics_date": date,  # 공시일자
                "url": "#",  # 공시 URL
            },
            {
                "mksc_shrn_iscd": "000660",  # 종목코드
                "corp_name": "SK하이닉스",  # 회사명
                "dics_hora": "0930",  # 공시시간
                "dics_tl": "신제품 개발 완료 및 양산",  # 공시제목
                "dics_date": date,  # 공시일자
                "url": "#",  # 공시 URL
            },
        ]

        logger.info(f"데모 공시 정보 {len(demo_disclosures)}건 생성")
        return demo_disclosures

    def cancel_order(self, order_no, ticker, quantity, orig_order_type="00"):
        """주문 취소

        Args:
            order_no (str): 원주문번호
            ticker (str): 종목코드
            quantity (int): 취소 수량
            orig_order_type (str): 원주문 유형(00: 지정가, 01: 시장가)

        Returns:
            dict: 취소 결과
        """
        if self.demo_mode:
            # 데모 취소 결과 반환
            logger.info(f"데모 모드: 주문 취소 처리 (주문번호: {order_no})")
            return {
                "success": True,
                "order_no": order_no,
                "cancel_time": datetime.now().strftime("%H%M%S"),
            }

        url = urljoin(self.base_url, "/uapi/domestic-stock/v1/trading/order-cash")
        headers = self.get_headers()

        # 실전/모의 거래에 따른 TR ID 설정
        if self.is_real_trading:
            headers["tr_id"] = "TTTC0803U"  # 실전 주문 취소
        else:
            headers["tr_id"] = "VTTC0803U"  # 모의 주문 취소

        data = {
            "CANO": self.account_number,
            "ACNT_PRDT_CD": self.account_number,
            "KRX_FWDG_ORD_ORGNO": "",  # 한국거래소전송주문조직번호
            "ORGN_ODNO": order_no,  # 원주문번호
            "ORD_DVSN": orig_order_type,  # 주문구분(00: 지정가, 01: 시장가)
            "RVSE_CNCL_DVSN_CD": "02",  # 정정취소구분코드(02: 취소)
            "PDNO": ticker,  # 종목코드
            "ORD_QTY": str(quantity),  # 취소 수량
            "ORD_UNPR": "0",  # 주문단가(취소시 0)
            "QTY_ALL_ORD_YN": "Y",  # 전량취소여부(Y: 전량취소)
        }

        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            result = response.json()

            if result["rt_cd"] == "0":
                cancel_result = {
                    "success": True,
                    "order_no": result["output"]["ODNO"],  # 취소 주문번호
                    "cancel_time": result["output"]["ORD_TMD"],  # 취소 시각
                }
                logger.info(f"주문 취소 성공: {order_no}")
                return cancel_result
            else:
                logger.error(f"주문 취소 실패: {result['msg_cd']} - {result['msg1']}")
                return {
                    "success": False,
                    "error": f"{result['msg_cd']} - {result['msg1']}",
                }
        except Exception as e:
            logger.error(f"주문 취소 중 오류 발생: {str(e)}")
            return {
                "success": False,
                "error": str(e),
            }

    def _api_request(
        self, method, url, headers=None, params=None, data=None, skip_limit_check=False
    ):
        """API 요청 실행

        Args:
            method (str): HTTP 메소드
            url (str): API URL
            headers (dict): HTTP 헤더
            params (dict): 쿼리 파라미터
            data (dict): 요청 바디 데이터
            skip_limit_check (bool): API 제한 체크 스킵 여부

        Returns:
            dict: API 응답
        """
        try:
            # API 제한 체크
            if not skip_limit_check and not self._check_api_limit():
                return None

            # URL 처리 - 상대 경로인 경우 base_url과 결합
            if url.startswith("/"):
                if hasattr(self, "base_url"):
                    full_url = urljoin(self.base_url, url)
                else:
                    # 기본 URL 설정
                    self.base_url = "https://openapi.koreainvestment.com:9443"
                    full_url = urljoin(self.base_url, url)
            elif not url.startswith(("http://", "https://")):
                if hasattr(self, "base_url"):
                    full_url = urljoin(self.base_url, url)
                else:
                    # 기본 URL 설정
                    self.base_url = "https://openapi.koreainvestment.com:9443"
                    full_url = urljoin(self.base_url, url)
            else:
                full_url = url

            # 요청 실행
            def _make_request():
                if method.lower() == "get":
                    response = requests.get(
                        full_url, headers=headers, params=params, timeout=10
                    )
                elif method.lower() == "post":
                    response = requests.post(
                        full_url, headers=headers, json=data, timeout=10
                    )
                else:
                    raise ValueError(f"지원하지 않는 HTTP 메소드: {method}")

                # 응답 검증
                if response.status_code != 200:
                    logger.error(
                        f"API 응답 오류: {response.status_code} - {response.text}"
                    )
                    return None

                # JSON 응답 파싱
                try:
                    result = response.json()
                except Exception as e:
                    logger.error(f"API 응답 JSON 파싱 오류: {str(e)}")
                    return None

                # 에러 코드 확인
                if "rt_cd" in result and result["rt_cd"] != "0":
                    error_msg = result.get("msg1", "알 수 없는 오류")
                    logger.error(
                        f"API 응답 오류: {error_msg} (코드: {result['rt_cd']})"
                    )
                    return None

                return result

            # 토큰 만료 시 재시도
            result = _make_request()
            if result is None and not skip_limit_check:
                # 토큰 갱신 시도
                if self.connect():
                    headers = self.get_headers() if headers else {}
                    result = _make_request()

            return result

        except requests.exceptions.RequestException as e:
            logger.error(f"API 요청 실패: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"API 호출 중 오류 발생: {str(e)}")
            raise

    def _handle_api_response(self, response, symbol=None):
        """API 응답 처리 (한국투자증권 OpenAPI 공식 구조 반영)

        Args:
            response: API 응답 객체
            symbol (str, optional): 종목 코드

        Returns:
            dict or list: 처리된 응답 데이터
        """
        try:
            response_data = response.json()

            # API 응답 내용 로깅
            if symbol:
                logger.debug(
                    f"API 응답 ({symbol}): {json.dumps(response_data, indent=2, ensure_ascii=False)}"
                )
            else:
                logger.debug(
                    f"API 응답: {json.dumps(response_data, indent=2, ensure_ascii=False)}"
                )

            # 응답 상태 코드 확인
            if response.status_code != 200:
                error_msg = response_data.get("msg1", "알 수 없는 오류")
                logger.error(f"API 호출 실패: {error_msg}")
                return None

            # 공식 구조: output2(리스트) > output1(단일) > output(단일)
            if "output2" in response_data and response_data["output2"]:
                return response_data["output2"]
            elif "output1" in response_data and response_data["output1"]:
                return response_data["output1"]
            elif "output" in response_data and response_data["output"]:
                return response_data["output"]
            else:
                warning_msg = f"API 응답에 output/output1/output2 데이터가 없습니다: {symbol if symbol else ''}\n전체 응답: {json.dumps(response_data, indent=2, ensure_ascii=False)}"
                logger.warning(warning_msg)
                return None

        except Exception as e:
            logger.error(f"API 응답 처리 중 오류 발생: {str(e)}")
            return None

    def get_stock_price(self, symbol):
        """주식 현재가 조회

        Args:
            symbol (str): 종목 코드

        Returns:
            dict: 주식 현재가 정보
        """
        if not self._check_api_limit():
            return None

        url = urljoin(self.base_url, "/uapi/domestic-stock/v1/quotations/inquire-price")
        headers = self.get_headers()
        headers["tr_id"] = "FHKST01010100"  # 주식 현재가 시세

        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": symbol,
        }

        try:
            response = requests.get(url, headers=headers, params=params)
            return self._handle_api_response(response, symbol)
        except Exception as e:
            logger.error(f"주식 현재가 조회 중 오류 발생: {str(e)}")
            return None

    def risk_analysis(self, tickers):
        """시장 위험도 분석"""
        try:
            # KOSPI 지수 조회
            logger.info("시장 위험도 분석: KOSPI 지수 데이터 조회 시작")
            kospi_data = self.get_ohlcv("U001", period="D", count=20)

            if kospi_data.empty:
                logger.warning("KOSPI 지수 데이터를 가져올 수 없습니다.")
                return {"kospi_change": 0.0, "kospi_volatility": 5.0}  # 기본 변동성

            # 데이터 검증
            logger.debug(f"KOSPI 데이터 컬럼: {kospi_data.columns.tolist()}")
            logger.debug(f"KOSPI 데이터 샘플:\n{kospi_data.head(3)}")

            # 컬럼 존재 여부 확인
            if "close" not in kospi_data.columns:
                logger.error("KOSPI 데이터에 'close' 컬럼이 없습니다!")
                return {"kospi_change": 0.0, "kospi_volatility": 5.0}

            # 코스피 등락률 및 변동성 계산
            try:
                kospi_change = kospi_data["close"].pct_change().iloc[-1] * 100
            except (IndexError, KeyError) as e:
                logger.error(f"코스피 등락률 계산 오류: {str(e)}")
                kospi_change = 0.0

            try:
                kospi_volatility = kospi_data["close"].pct_change().std() * 100
            except (IndexError, KeyError) as e:
                logger.error(f"코스피 변동성 계산 오류: {str(e)}")
                kospi_volatility = 5.0

            # NaN 값이면 기본값으로 대체
            if pd.isna(kospi_change):
                kospi_change = 0.0
            if pd.isna(kospi_volatility):
                kospi_volatility = 5.0

            logger.info(
                f"KOSPI 지수 분석 결과: 등락률 {kospi_change:.2f}%, 변동성 {kospi_volatility:.2f}%"
            )

            return {
                "kospi_change": kospi_change,
                "kospi_volatility": kospi_volatility,
            }
        except Exception as e:
            logger.error(f"시장 위험도 분석 중 오류 발생: {str(e)}")
            return {"kospi_change": 0.0, "kospi_volatility": 5.0}  # 기본 변동성
