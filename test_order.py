import logging
import sys
from api.kis_api import KisAPI

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("test_order")


def test_order_functions():
    """주문 함수 테스트"""
    logger.info("=== 주문 기능 테스트 시작 ===")

    # KisAPI 클래스 초기화
    api = KisAPI()

    # 연결 상태 확인
    logger.info(f"API 연결 상태: {api.is_connected}")
    if not api.is_connected:
        logger.error("API 서버에 연결되지 않았습니다. 테스트를 중단합니다.")
        return

    # 데모 모드 확인
    logger.info(f"데모 모드 설정: {api.demo_mode}")

    # 계좌 정보 조회
    account_info = api.get_account_info()
    logger.info(f"계좌 잔고: {account_info['balance']}원")
    logger.info(f"총 보유 자산 가치: {account_info['total_value']}원")

    # 테스트할 종목 - 삼성전자
    test_ticker = "005930"

    # 현재가 조회
    current_price = api.get_current_price(test_ticker)
    logger.info(f"삼성전자(005930) 현재가: {current_price}원")

    # 호가단위 테스트
    tick_size = api.get_tick_size(current_price)
    logger.info(f"현재가 {current_price}원의 호가단위: {tick_size}원")

    # 호가단위 반올림 테스트
    test_prices = [
        current_price - tick_size * 0.5,  # 내림되어야 함
        current_price,  # 그대로
        current_price + tick_size * 0.4,  # 내림되어야 함
        current_price + tick_size * 0.5,  # 올림되어야 함
    ]

    logger.info("호가단위 반올림 테스트:")
    for price in test_prices:
        rounded_price = api.round_to_tick_size(price)
        logger.info(f"원래 가격: {price:.1f}원 -> 호가단위 조정: {rounded_price}원")

    # 실제 주문은 실행하지 않고 API 호출 형태만 테스트
    # 시장가 매수 테스트 (실제 주문하지 않음)
    logger.info("시장가 매수 함수 호출 (테스트 모드: 실제 주문 실행 안함)")
    # api.market_buy(test_ticker, 1)  # 실제 주문 실행을 원하면 주석 해제

    # 지정가 매수 테스트 (실제 주문하지 않음)
    logger.info("지정가 매수 함수 호출 (테스트 모드: 실제 주문 실행 안함)")
    # 현재가보다 5% 낮은 가격으로 지정가 매수 테스트
    target_price = int(current_price * 0.95)
    adjusted_price = api.round_to_tick_size(target_price)
    logger.info(f"원래 목표가: {target_price}원 -> 호가단위 조정: {adjusted_price}원")
    # api.limit_buy(test_ticker, 1, target_price)  # 실제 주문 실행을 원하면 주석 해제

    # 시장가 매도 테스트 (실제 주문하지 않음)
    logger.info("시장가 매도 함수 호출 (테스트 모드: 실제 주문 실행 안함)")
    # api.market_sell(test_ticker, 1)  # 실제 주문 실행을 원하면 주석 해제

    # 지정가 매도 테스트 (실제 주문하지 않음)
    logger.info("지정가 매도 함수 호출 (테스트 모드: 실제 주문 실행 안함)")
    # 현재가보다 5% 높은 가격으로 지정가 매도 테스트
    target_price = int(current_price * 1.05)
    adjusted_price = api.round_to_tick_size(target_price)
    logger.info(f"원래 목표가: {target_price}원 -> 호가단위 조정: {adjusted_price}원")
    # api.limit_sell(test_ticker, 1, target_price)  # 실제 주문 실행을 원하면 주석 해제

    logger.info("주문 함수 호출 테스트 완료")

    # 주문 관련 코드 검사
    try:
        # 실제 주문을 실행하지 않고 메서드만 내부 동작 확인
        # place_order 메서드의 로직 확인
        logger.info("place_order 메서드 내부 검사")
        result = api._check_api_limit()
        logger.info(f"API 호출 제한 체크 결과: {result}")

        # 데모 모드 여부 재확인
        logger.info(f"데모 모드 설정: {api.demo_mode} (False이면 실제 거래 모드)")
        if api.demo_mode:
            logger.warning(
                "데모 모드가 활성화되어 있어서 실제 주문은 진행되지 않습니다!"
            )
        else:
            logger.info("실제 거래 모드 상태입니다.")

        logger.info("주문 함수 테스트 성공!")
    except Exception as e:
        logger.error(f"주문 함수 테스트 중 오류 발생: {str(e)}")


if __name__ == "__main__":
    logger.info("주문 기능 테스트 스크립트 시작")

    try:
        test_order_functions()
        logger.info("모든 테스트가 완료되었습니다.")
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {str(e)}")
        raise
