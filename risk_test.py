#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import argparse
from datetime import datetime

import config
from api.kis_api import KisAPI
from risk.risk_manager import RiskManager
from screener.stock_screener import StockScreener


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description="리스크 관리 모듈 테스트 스크립트")

    # 실행 모드 선택
    parser.add_argument(
        "--mode",
        type=str,
        choices=["assess", "optimize", "report", "stoploss"],
        default="assess",
        help="실행 모드 - assess: 시장 리스크 평가, optimize: 포트폴리오 최적화, report: 리스크 보고서 생성, stoploss: 동적 손절/익절 테스트",
    )

    # 동적 손절/익절 테스트용 종목 코드
    parser.add_argument(
        "--ticker", type=str, help="종목 코드 (stoploss 모드일 때 필요)"
    )

    # 포트폴리오 설정
    parser.add_argument(
        "--portfolio-value", type=int, default=10000000, help="포트폴리오 평가금액 (원)"
    )

    # 로깅 관련 인자
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="로그 레벨",
    )

    # 데모 모드 설정
    parser.add_argument(
        "--demo", action="store_true", help="API 연결 없이 더미 데이터로 테스트"
    )

    return parser.parse_args()


def main():
    """리스크 테스트 실행 함수"""
    # 명령행 인자 파싱
    args = parse_args()

    # 로깅 설정
    log_level = getattr(logging, args.log_level)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                f'logs/risk_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
                encoding="utf-8",
            ),
        ],
    )
    logger = logging.getLogger("risk_test")

    # 로그 디렉토리 생성
    os.makedirs("logs", exist_ok=True)

    # 리스크 테스트 시작 로그
    logger.info(f"리스크 관리 테스트 시작 - 모드: {args.mode}")

    try:
        # API 클라이언트 초기화
        if args.demo:
            logger.info("데모 모드로 실행 (API 연결 없음)")
            api_client = None
        else:
            api_client = KisAPI(
                app_key=config.APP_KEY,
                app_secret=config.APP_SECRET,
                account_number=config.ACCOUNT_NUMBER,
                account_code=config.ACCOUNT_CODE,
                is_demo=config.IS_DEMO_ACCOUNT,
            )

        # RiskManager 초기화
        risk_manager = RiskManager(api_client)

        # 모드에 따른 실행
        if args.mode == "assess":
            # 시장 리스크 평가
            logger.info("시장 리스크 평가 실행")
            risk_manager.assess_market_risk()

            # 결과 출력
            logger.info(f"현재 리스크 상태: {risk_manager.risk_status}")
            logger.info(f"현재 시장 상태: {risk_manager.market_condition}")

            # 투자 비중 조정계수 출력
            position_multiplier = risk_manager.position_size_multiplier[
                risk_manager.risk_status
            ]
            logger.info(
                f"투자 비중 조정계수: {position_multiplier} (기준 대비 {position_multiplier*100}%)"
            )

            # 손절/익절 조정계수 출력
            loss_cut_multiplier = 1.0  # 손절 비율은 조정하지 않음
            profit_cut_multiplier = risk_manager.profit_cut_multiplier[
                risk_manager.risk_status
            ]
            logger.info(
                f"익절 비율 조정계수: {profit_cut_multiplier} (기준 대비 {profit_cut_multiplier*100}%)"
            )

        elif args.mode == "optimize":
            # 종목 선별기 초기화
            screener = StockScreener(api_client)

            # 후보 종목 가져오기
            logger.info("후보 종목 스크리닝")
            candidates = screener.run_screening(market_list=["KOSPI", "KOSDAQ"])
            top_candidates = screener.get_final_candidates(limit=10)

            if not top_candidates:
                logger.warning("후보 종목이 없습니다.")
                return 1

            logger.info(f"후보 종목 {len(top_candidates)}개 발견")

            # 포트폴리오 최적 배분 계산
            logger.info("포트폴리오 최적 배분 계산")
            allocation = risk_manager.calculate_optimal_portfolio_allocation(
                top_candidates, args.portfolio_value
            )

            # 결과 출력
            logger.info("최적 포트폴리오 배분:")
            for ticker, info in allocation.items():
                logger.info(
                    f"종목: {ticker}, 비중: {info['weight']*100:.2f}%, 금액: {info['amount']:,.0f}원"
                )

        elif args.mode == "report":
            # 리스크 보고서 생성
            logger.info("리스크 보고서 생성")
            report_file = risk_manager.generate_risk_report()

            if report_file:
                logger.info(f"리스크 보고서가 생성되었습니다: {report_file}")

                # 보고서 내용 출력
                try:
                    with open(report_file, "r", encoding="utf-8") as f:
                        report_content = f.read()
                    logger.info(f"보고서 내용:\n{report_content}")
                except Exception as e:
                    logger.error(f"보고서 읽기 실패: {e}")
            else:
                logger.error("리스크 보고서 생성에 실패했습니다.")

        elif args.mode == "stoploss":
            # 동적 손절/익절 테스트
            if not args.ticker:
                logger.error("stoploss 모드에서는 --ticker 인자가 필요합니다.")
                return 1

            ticker = args.ticker
            logger.info(f"종목 {ticker}에 대한 동적 손절/익절 테스트")

            # 현재가 조회
            try:
                if api_client:
                    stock_info = api_client.get_stock_price(ticker)
                    current_price = float(stock_info["output"]["stck_prpr"])
                    logger.info(f"현재가: {current_price:,.0f}원")
                else:
                    # 데모 모드에서는 임의의 가격 사용
                    current_price = 50000
                    logger.info(f"데모 모드 현재가: {current_price:,.0f}원")

                # 동적 손절가 계산
                stoploss = risk_manager.calculate_dynamic_stoploss(
                    ticker, current_price
                )

                # 동적 익절가 계산
                takeprofit = risk_manager.calculate_dynamic_takeprofit(
                    ticker, current_price
                )

                # 결과 출력
                logger.info(f"종목: {ticker}")
                logger.info(f"매수가: {current_price:,.0f}원")
                logger.info(
                    f"동적 손절가: {stoploss:,.0f}원 (매수가 대비 {(stoploss/current_price-1)*100:.2f}%)"
                )
                logger.info(
                    f"동적 익절가: {takeprofit:,.0f}원 (매수가 대비 {(takeprofit/current_price-1)*100:.2f}%)"
                )
                logger.info(
                    f"리스크/리워드 비율: {(takeprofit-current_price)/(current_price-stoploss):.2f}"
                )

            except Exception as e:
                logger.error(f"동적 손절/익절 계산 중 오류 발생: {e}")
                import traceback

                logger.error(traceback.format_exc())
                return 1

        logger.info("리스크 관리 테스트 완료")
        return 0

    except Exception as e:
        logger.error(f"리스크 테스트 실행 중 오류 발생: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
