import logging
import sys
from datetime import datetime, timedelta
from news_analyzer import NewsAnalyzer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("test_risk_management")


def test_risk_management():
    """리스크 관리 기능 테스트"""
    logger.info("리스크 관리 기능 테스트 시작")

    analyzer = NewsAnalyzer()

    # 1. 포지션 크기 계산 테스트
    ticker = "005930"  # 삼성전자
    price = 70000
    total_capital = 10000000  # 1000만원

    quantity = analyzer.calculate_position_size(ticker, price, total_capital)
    logger.info(f"종목: {ticker}, 가격: {price}원, 총 자본금: {total_capital}원")
    logger.info(f"계산된 매수 수량: {quantity}주 (투자금액: {quantity * price}원)")

    # 2. 동적 손절가 계산 테스트
    entry_price = 70000
    current_price = 68000
    position_age = 1

    stop_loss_price = analyzer.calculate_dynamic_stop_loss(
        entry_price, current_price, position_age
    )
    logger.info(
        f"진입가: {entry_price}원, 현재가: {current_price}원, 보유일수: {position_age}일"
    )
    logger.info(
        f"계산된 손절가: {stop_loss_price}원 (손실률: {(stop_loss_price - entry_price) / entry_price:.2%})"
    )

    # 3. 트레일링 스탑 계산 테스트
    entry_price = 70000
    current_price = 75000
    highest_price = 77000

    trailing_stop = analyzer.calculate_trailing_stop(
        entry_price, current_price, highest_price
    )
    logger.info(
        f"진입가: {entry_price}원, 현재가: {current_price}원, 최고가: {highest_price}원"
    )
    logger.info(
        f"계산된 트레일링 스탑: {trailing_stop}원 (최고가 대비: {(trailing_stop - highest_price) / highest_price:.2%})"
    )

    # 4. 포지션 청산 여부 테스트
    scenarios = [
        # 익절 시나리오
        {
            "ticker": "005930",
            "entry_price": 70000,
            "current_price": 73500,  # +5% 이상
            "position_age": 1,
            "highest_price": 73500,
            "desc": "익절 조건",
        },
        # 손절 시나리오
        {
            "ticker": "005930",
            "entry_price": 70000,
            "current_price": 68000,  # -약 3%
            "position_age": 1,
            "highest_price": 70500,
            "desc": "손절 조건",
        },
        # 트레일링 스탑 시나리오
        {
            "ticker": "005930",
            "entry_price": 70000,
            "current_price": 71500,  # 고점 대비 하락
            "position_age": 2,
            "highest_price": 74000,  # 고점 대비 -3.4%
            "desc": "트레일링 스탑 조건",
        },
        # 최대 보유 기간 초과 시나리오
        {
            "ticker": "005930",
            "entry_price": 70000,
            "current_price": 71000,
            "position_age": 3,
            "highest_price": 71500,
            "desc": "최대 보유 기간 조건",
        },
    ]

    logger.info("포지션 청산 판단 테스트")
    for scenario in scenarios:
        decision = analyzer.should_close_position(
            scenario["ticker"],
            scenario["entry_price"],
            scenario["current_price"],
            scenario["position_age"],
            scenario["highest_price"],
        )

        logger.info(f"시나리오: {scenario['desc']}")
        logger.info(
            f"청산 여부: {decision['close']}, 이유: {decision['reason']}, 수익률: {decision['profit_loss']:.2%}"
        )

    # 5. 일일 손실 한도 테스트
    positions = [
        {
            "ticker": "005930",
            "entry_price": 70000,
            "current_price": 67000,  # -4.3%
            "quantity": 30,
        },
        {
            "ticker": "000660",
            "entry_price": 120000,
            "current_price": 118000,  # -1.7%
            "quantity": 20,
        },
        {
            "ticker": "035420",
            "entry_price": 350000,
            "current_price": 347000,  # -0.9%
            "quantity": 10,
        },
    ]

    risk_check = analyzer.check_daily_risk_limit(positions)
    logger.info("일일 손실 한도 테스트")
    logger.info(f"손실 한도 초과 여부: {risk_check['exceeded']}")
    logger.info(f"현재 손익률: {risk_check['current_loss']:.2%}")
    if risk_check["exceeded"]:
        logger.info(f"메시지: {risk_check['message']}")

    logger.info("리스크 관리 기능 테스트 완료")


def test_backtest():
    """백테스팅 기능 테스트"""
    logger.info("백테스팅 기능 테스트 시작")

    analyzer = NewsAnalyzer()

    # 테스트 종목 코드
    test_tickers = [
        "005930",  # 삼성전자
        "000660",  # SK하이닉스
        "035420",  # NAVER
        "035720",  # 카카오
        "051910",  # LG화학
        "000270",  # 기아
        "068270",  # 셀트리온
        "105560",  # KB금융
        "055550",  # 신한지주
        "036570",  # 엔씨소프트
    ]

    # 테스트 기간 (1개월)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    # 백테스팅 실행
    results = analyzer.run_backtest(
        strategy_name="뉴스 기반 단기 매매 테스트",
        tickers=test_tickers,
        start_date=start_date,
        end_date=end_date,
        initial_capital=10000000,  # 1000만원
    )

    # 결과 요약 출력
    logger.info("백테스팅 결과 요약")
    logger.info(f"전략: {results['strategy_name']}")
    logger.info(f"기간: {results['start_date']} ~ {results['end_date']}")
    logger.info(f"초기 자본금: {results['initial_capital']:,.0f}원")
    logger.info(f"최종 자본금: {results['final_capital']:,.0f}원")
    logger.info(f"총 수익률: {results['total_return']:.2f}%")
    logger.info(f"연간 수익률: {results['annual_return']:.2f}%")
    logger.info(f"최대 낙폭(MDD): {results['max_drawdown']*100:.2f}%")
    logger.info(f"승률: {results['win_rate']:.2f}%")
    logger.info(f"총 거래 횟수: {len(results['trades'])}")

    # 상위 3개 거래 출력
    if results["trades"]:
        logger.info("주요 거래 내역:")
        # 수익 기준 정렬
        sorted_trades = sorted(
            results["trades"], key=lambda x: x["profit"], reverse=True
        )
        for i, trade in enumerate(sorted_trades[:3]):
            logger.info(
                f"거래 #{i+1}: {trade['ticker']}, 매수가: {trade['entry_price']:,.0f}원, 매도가: {trade['exit_price']:,.0f}원"
            )
            logger.info(
                f"  수익: {trade['profit']:,.0f}원 ({trade['profit_rate']*100:.2f}%), 보유기간: {trade['holding_period']}일"
            )
            logger.info(f"  청산이유: {trade['exit_reason']}")

    logger.info("백테스팅 기능 테스트 완료")


if __name__ == "__main__":
    logger.info("리스크 관리 및 백테스팅 기능 테스트 시작")
    test_risk_management()
    print("\n" + "=" * 80 + "\n")
    test_backtest()
    logger.info("테스트 완료")
