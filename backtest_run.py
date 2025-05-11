#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta

import config
from api.kis_api import KisAPI
from backtesting.backtester import BackTester


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description="백테스팅 실행 스크립트")

    # 날짜 인자 (기본값: 현재 날짜 기준 90일 전 ~ 현재)
    default_end_date = datetime.now().strftime("%Y%m%d")
    default_start_date = (datetime.now() - timedelta(days=90)).strftime("%Y%m%d")

    parser.add_argument(
        "--start-date",
        type=str,
        default=default_start_date,
        help="백테스팅 시작일 (YYYYMMDD 형식)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=default_end_date,
        help="백테스팅 종료일 (YYYYMMDD 형식)",
    )

    # 자본금 및 투자 비율 관련 인자
    parser.add_argument(
        "--capital", type=int, default=10000000, help="초기 자본금 (원)"
    )
    parser.add_argument(
        "--profit-cut", type=float, default=0.05, help="익절 비율 (0.05 = 5 퍼센트)"
    )
    parser.add_argument(
        "--loss-cut", type=float, default=0.02, help="손절 비율 (0.02 = 2 퍼센트)"
    )
    parser.add_argument("--max-stocks", type=int, default=3, help="최대 보유 종목 수")

    # 로깅 관련 인자
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="로그 레벨",
    )

    # 결과 저장 관련 인자
    parser.add_argument(
        "--report-dir", type=str, default="data/backtest", help="결과 저장 디렉토리"
    )

    return parser.parse_args()


def main():
    """백테스팅 실행 함수"""
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
                f'logs/backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
                encoding="utf-8",
            ),
        ],
    )
    logger = logging.getLogger("backtest_run")

    # 로그 디렉토리 생성
    os.makedirs("logs", exist_ok=True)

    # 백테스팅 시작 로그
    logger.info("백테스팅 실행 시작")
    logger.info(
        f"설정: 시작일={args.start_date}, 종료일={args.end_date}, 자본금={args.capital:,}원"
    )
    logger.info(
        f"설정: 익절={args.profit_cut*100:.0f}%, 손절={args.loss_cut*100:.0f}%, 최대종목수={args.max_stocks}"
    )

    try:
        # 설정 업데이트
        config.PROFIT_CUT_RATIO = args.profit_cut
        config.LOSS_CUT_RATIO = args.loss_cut
        config.MAX_STOCK_COUNT = args.max_stocks

        # KisAPI 클라이언트 초기화 (백테스팅은 데모 모드로 실행)
        api_client = KisAPI(
            app_key=config.APP_KEY,
            app_secret=config.APP_SECRET,
            account_number=config.ACCOUNT_NUMBER,
            is_demo=True,  # 백테스팅은 항상 데모 모드
        )

        # 날짜 형식 확인 및 변환
        start_date = args.start_date.replace("-", "")
        end_date = args.end_date.replace("-", "")

        # BackTester 초기화 및 실행
        backtester = BackTester(
            api_client=api_client, start_date=start_date, end_date=end_date
        )

        # 초기 자본금 설정
        backtester.initial_capital = args.capital

        # 백테스팅 실행
        results = backtester.run_backtest()

        # 결과 분석 및 저장
        performance = backtester._analyze_backtest_results()

        # 성능 지표 출력
        logger.info("=" * 50)
        logger.info("백테스팅 결과 요약")
        logger.info("=" * 50)
        logger.info(f"총 거래 횟수: {performance['total_trades']}회")
        logger.info(f"초기 자본금: {args.capital:,}원")
        logger.info(f"최종 자산: {performance['final_capital']:,.0f}원")
        logger.info(f"총 수익률: {performance['total_return']:.2f}%")
        logger.info(f"연환산 수익률: {performance['annual_return']:.2f}%")
        logger.info(f"승률: {performance['win_rate']:.2f}%")
        logger.info(f"최대 낙폭(MDD): {performance['max_drawdown']:.2f}%")
        logger.info(f"샤프 비율: {performance['sharpe_ratio']:.2f}")
        logger.info("=" * 50)

        # 성능 그래프 저장
        report_path = os.path.join(
            args.report_dir,
            f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        backtester.plot_performance(f"{report_path}.png")

        logger.info(f"백테스팅 결과가 {report_path}.png에 저장되었습니다.")
        return 0

    except Exception as e:
        logger.error(f"백테스팅 실행 중 오류 발생: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
