import logging
import sys
import pandas as pd
from api.kis_api import KisAPI

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("test_kis_api")


def test_api_connection():
    """API 연결 테스트"""
    logger.info("=== API 연결 테스트 시작 ===")

    # KisAPI 클래스 초기화 (데모 모드 파라미터 전달해도 무시됨)
    api = KisAPI(demo_mode=True)

    # 데모 모드 설정 확인
    logger.info(f"데모 모드 설정: {api.demo_mode}")
    assert api.demo_mode == False, "데모 모드가 여전히 활성화되어 있습니다!"

    # 연결 상태 확인
    logger.info(f"API 연결 상태: {api.is_connected}")
    assert api.is_connected, "API 서버에 연결되지 않았습니다!"

    logger.info("API 연결 테스트 성공!")
    return api


def test_get_ohlcv(api, ticker="005930"):
    """OHLCV 데이터 조회 테스트"""
    logger.info(f"=== OHLCV 데이터 조회 테스트 시작 (종목: {ticker}) ===")

    # 주식 데이터 조회
    df = api.get_ohlcv(ticker, count=10)

    logger.info(f"조회된 데이터 개수: {len(df)}")
    logger.info(f"데이터 컬럼: {df.columns.tolist()}")

    # 필수 컬럼 확인
    required_columns = ["date", "open", "high", "low", "close", "volume"]
    for col in required_columns:
        assert col in df.columns, f"필수 컬럼 '{col}'이 없습니다!"

    # 데이터 출력
    logger.info(f"데이터 샘플:\n{df.head(3)}")

    logger.info("OHLCV 데이터 조회 테스트 성공!")
    return df


def test_get_index_data(api, ticker="U001"):
    """지수 데이터 조회 테스트"""
    logger.info(f"=== 지수 데이터 조회 테스트 시작 (종목: {ticker}) ===")

    # 지수 데이터 조회
    df = api.get_ohlcv(ticker, count=10)

    logger.info(f"조회된 데이터 개수: {len(df)}")
    logger.info(f"데이터 컬럼: {df.columns.tolist()}")

    # 필수 컬럼 확인
    required_columns = ["date", "open", "high", "low", "close"]
    for col in required_columns:
        assert col in df.columns, f"필수 컬럼 '{col}'이 없습니다!"

    # 데이터 출력
    logger.info(f"데이터 샘플:\n{df.head(3)}")

    logger.info("지수 데이터 조회 테스트 성공!")
    return df


def test_fallback_mechanisms(api):
    """대체 로직 테스트"""
    logger.info("=== 대체 로직 테스트 시작 ===")

    # 코스닥 지수 조회 (대체 로직이 작동할 가능성 높음)
    logger.info("코스닥 지수 데이터 조회 시도...")
    kosdaq_df = api.get_ohlcv("U201", count=5)
    logger.info(f"코스닥 데이터 조회 결과: {len(kosdaq_df)}개 데이터")

    # 존재하지 않는 지수 코드로 테스트 (ETF 대체 로직 확인)
    logger.info("존재하지 않는 지수 코드로 조회 시도...")
    try:
        test_df = api.get_ohlcv("U999", count=5)
        logger.info(f"존재하지 않는 지수 조회 결과: {len(test_df)}개 데이터")
    except Exception as e:
        logger.info(f"예상된 오류 발생: {str(e)}")

    logger.info("대체 로직 테스트 완료!")


if __name__ == "__main__":
    try:
        # API 연결 테스트
        api = test_api_connection()

        # 주식 데이터 조회 테스트
        test_get_ohlcv(api, "005930")  # 삼성전자

        # 지수 데이터 조회 테스트
        test_get_index_data(api, "U001")  # KOSPI

        # 대체 로직 테스트
        test_fallback_mechanisms(api)

        logger.info("모든 테스트가 성공적으로 완료되었습니다!")

    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {str(e)}")
        raise
