import logging
import os
from news_analyzer import NewsAnalyzer
from api.kis_api import KisAPI
import config

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_news_analyzer")


def test_news_and_theme_analysis():
    """
    뉴스와 테마 분석 기능을 테스트합니다.
    """
    logger.info("뉴스와 테마 분석 기능 테스트 시작")

    # KIS API 클라이언트 초기화
    try:
        # 데모 모드로 설정
        demo_mode = True  # True로 설정하면 실제 거래 없이 테스트만 가능

        api_client = KisAPI(
            app_key=config.APP_KEY,
            app_secret=config.APP_SECRET,
            account_number=config.ACCOUNT_NUMBER,
            demo_mode=demo_mode,
        )

        # 토큰 발급 명시적으로 시도
        is_token_issued = api_client.issue_access_token()
        if is_token_issued:
            logger.info("KIS API 토큰 발급 성공")
        else:
            logger.error("KIS API 토큰 발급 실패, 데모 데이터로 대체합니다.")
            # 토큰 발급 실패 시 데모 모드 강제 설정
            api_client.set_demo_mode(True)

        logger.info("KIS API 클라이언트 초기화 성공")
    except Exception as e:
        logger.error(f"KIS API 클라이언트 초기화 실패: {str(e)}")
        logger.info("더미 데이터를 사용하여 테스트를 진행합니다.")
        api_client = None

    # NewsAnalyzer 인스턴스 생성 (API 클라이언트 전달)
    analyzer = NewsAnalyzer(api_client=api_client)

    # 마스터 데이터 로드
    analyzer.load_master_data()

    # 필터링 테스트 (KOSPI, 시가총액 100억 이상)
    filtered_stocks = analyzer.get_filtered_stocks(market="KOSPI", cap_min=100)
    logger.info(f"필터링된 종목 수: {len(filtered_stocks)}")
    if filtered_stocks:
        logger.info(f"필터링된 샘플 종목: {filtered_stocks[:3]}")

    # 테마 추출 테스트
    test_text = "삼성전자와 SK하이닉스는 반도체 업종의 대표 기업입니다. 전기차 시장에서는 LG화학과 삼성SDI가 배터리 기술력을 인정받고 있습니다."
    themes = analyzer.extract_themes(test_text)
    logger.info(f"추출된 테마: {themes}")

    # 실제 주가 정보 테스트
    if api_client and filtered_stocks:
        test_stock = filtered_stocks[0]
        price_info = analyzer.get_price_info(test_stock)
        if price_info:
            logger.info(f"종목 {test_stock} 주가 정보: {price_info}")
        else:
            logger.warning(f"종목 {test_stock} 주가 정보를 가져오지 못했습니다.")
            logger.info("더미 데이터를 사용하여 테스트를 계속 진행합니다.")

    # 종목 선정 테스트 (실행 시간이 길 수 있음)
    try:
        selected_stocks = analyzer.select_stocks_by_news_and_theme(
            news_days=1, market_cap_min=100, top_n=5, markets=["KOSPI", "KOSDAQ"]
        )
        logger.info(f"선정된 종목 수: {len(selected_stocks)}")

        # 선정된 종목 주가 정보 출력
        if selected_stocks:
            logger.info("선정된 종목 주가 정보:")
            for idx, stock in enumerate(selected_stocks[:3], 1):
                price_info = analyzer.get_price_info(stock["code"])
                if price_info:
                    logger.info(
                        f"{idx}. [{stock['code']}] {stock['name']} - 현재가: {price_info.get('close', 0):,}원, 전일대비: {price_info.get('change_ratio', 0):.2f}%"
                    )
                else:
                    logger.info(
                        f"{idx}. [{stock['code']}] {stock['name']} - 주가 정보 없음"
                    )

        # 보고서 생성 테스트
        if selected_stocks:
            report_path = analyzer.generate_stock_selection_report(selected_stocks)
            logger.info(f"생성된 보고서 경로: {report_path}")
        else:
            logger.warning("선정된 종목이 없어 보고서를 생성할 수 없습니다.")
    except Exception as e:
        logger.error(f"종목 선정 중 오류 발생: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())

    logger.info("뉴스와 테마 분석 기능 테스트 완료")


if __name__ == "__main__":
    test_news_and_theme_analysis()
