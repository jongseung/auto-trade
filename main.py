#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta

import config
from trader.auto_trader import AutoTrader


def parse_args():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(
        description="한국투자증권 API를 이용한 자동 매매 시스템"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="run",
        choices=["run", "once", "backtest"],
        help="실행 모드 (run: 자동 매매 실행, once: 1회 실행, backtest: 백테스트)",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        help="백테스트 시작일 (YYYY-MM-DD)",
        default=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
    )

    parser.add_argument(
        "--end-date",
        type=str,
        help="백테스트 종료일 (YYYY-MM-DD)",
        default=datetime.now().strftime("%Y-%m-%d"),
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="로그 레벨",
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="데모 모드로 실행 (API 연결 없이 테스트 실행)",
    )

    parser.add_argument(
        "--risk-assessment",
        action="store_true",
        help="시장 리스크 평가만 실행",
    )

    return parser.parse_args()


def main():
    """메인 함수"""
    # 명령행 인수 파싱
    args = parse_args()

    # 로깅 설정
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = config.setup_logging()

    logger.info("자동 매매 시스템을 시작합니다.")

    try:
        # 자동 매매 인스턴스 생성
        auto_trader = AutoTrader(demo_mode=args.demo)

        # 리스크 평가만 실행하는 경우
        if args.risk_assessment:
            logger.info("시장 리스크 평가를 실행합니다.")
            auto_trader.risk_manager.assess_market_risk()
            risk_report = auto_trader.risk_manager.generate_risk_report()
            logger.info(f"리스크 평가 완료: {risk_report}")
            return

        # 실행 모드에 따라 동작
        if args.mode == "run":
            logger.info("자동 매매 실행 모드를 시작합니다.")
            auto_trader.start()

            # schedule 라이브러리는 무한 루프에서 run_pending()을 호출해야 함
            try:
                import time
                import schedule

                logger.info("스케줄러 실행 중...")
                while auto_trader.running:
                    schedule.run_pending()
                    time.sleep(1)  # 1초마다 작업 확인
            except KeyboardInterrupt:
                logger.info("사용자에 의해 스케줄러가 중지되었습니다.")
                auto_trader.stop()

        elif args.mode == "once":
            logger.info("1회 실행 모드를 시작합니다.")
            auto_trader.run_once()

        elif args.mode == "backtest":
            logger.info(
                f"백테스트 모드를 시작합니다. (기간: {args.start_date} ~ {args.end_date})"
            )
            auto_trader.run_backtest(args.start_date, args.end_date)

    except KeyboardInterrupt:
        logger.info("사용자에 의해 프로그램이 종료되었습니다.")

    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())

    finally:
        logger.info("자동 매매 시스템을 종료합니다.")


if __name__ == "__main__":
    main()
