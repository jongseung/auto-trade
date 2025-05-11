import logging
import sys
from api.kis_api import KisAPI

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("test_tick_size")


def test_tick_size():
    """호가단위 함수 테스트"""
    logger.info("=== 호가단위 함수 테스트 시작 ===")

    # KisAPI 클래스 초기화
    api = KisAPI()

    # 다양한 가격대에서 호가단위 확인
    test_prices = [
        1500,  # 1원 단위
        3000,  # 5원 단위
        10000,  # 10원 단위
        30000,  # 50원 단위
        100000,  # 100원 단위
        300000,  # 500원 단위
        1000000,  # 1000원 단위
    ]

    logger.info("가격별 호가단위 테스트:")
    for price in test_prices:
        tick_size = api.get_tick_size(price)
        logger.info(f"가격: {price:,}원 -> 호가단위: {tick_size}원")

    logger.info("\n호가단위 반올림 테스트:")
    test_round_prices = [
        (1499, 1499),  # 1원 단위 - 그대로
        (3002, 3000),  # 5원 단위 - 5원 단위 내림
        (3003, 3005),  # 5원 단위 - 5원 단위 올림
        (10004, 10000),  # 10원 단위 - 10원 단위 내림
        (10005, 10010),  # 10원 단위 - 10원 단위 올림
        (30024, 30000),  # 50원 단위 - 50원 단위 내림
        (30025, 30050),  # 50원 단위 - 50원 단위 올림
        (100049, 100000),  # 100원 단위 - 100원 단위 내림
        (100050, 100100),  # 100원 단위 - 100원 단위 올림
        (300249, 300000),  # 500원 단위 - 500원 단위 내림
        (300250, 300500),  # 500원 단위 - 500원 단위 올림
        (1000499, 1000000),  # 1000원 단위 - 1000원 단위 내림
        (1000500, 1001000),  # 1000원 단위 - 1000원 단위 올림
    ]

    for original, expected in test_round_prices:
        rounded = api.round_to_tick_size(original)
        result = "성공" if rounded == expected else f"실패 (예상: {expected})"
        logger.info(f"가격: {original:,}원 -> 호가단위 조정: {rounded:,}원 -> {result}")

    logger.info("호가단위 함수 테스트 완료!")


if __name__ == "__main__":
    test_tick_size()
