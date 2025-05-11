import unittest
from datetime import datetime, timedelta
import logging
import os
import sys
import pandas as pd

# 상위 디렉토리 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trader.auto_trader import AutoTrader
from api.kis_api import KisAPI
from screener.stock_screener import StockScreener
from strategy.trading_strategy import TradingStrategy
from risk.risk_manager import RiskManager


class TestAutoTrader(unittest.TestCase):
    """자동매매 시스템 테스트 클래스"""

    @classmethod
    def setUpClass(cls):
        """테스트 클래스 초기화"""
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        cls.logger = logging.getLogger("test_auto_trader")

    def setUp(self):
        """각 테스트 케이스 초기화"""
        self.auto_trader = AutoTrader(demo_mode=True)
        self.logger.info("테스트 케이스 시작")

    def tearDown(self):
        """각 테스트 케이스 정리"""
        if hasattr(self, "auto_trader"):
            self.auto_trader.stop()
        self.logger.info("테스트 케이스 종료")

    def test_initialization(self):
        """초기화 테스트"""
        self.assertIsNotNone(self.auto_trader.api_client)
        self.assertIsNotNone(self.auto_trader.screener)
        self.assertIsNotNone(self.auto_trader.risk_manager)
        self.assertIsNotNone(self.auto_trader.strategy)
        self.assertFalse(self.auto_trader.running)
        self.assertEqual(len(self.auto_trader.candidate_stocks), 0)

    def test_news_analysis(self):
        """뉴스 분석 테스트"""
        self.auto_trader._run_news_analysis()
        self.assertIsNotNone(self.auto_trader.news_candidates)
        self.logger.info(
            f"뉴스 분석 결과: {len(self.auto_trader.news_candidates)}개 종목"
        )

    def test_screening(self):
        """스크리닝 테스트"""
        self.auto_trader._run_screening()
        self.assertIsNotNone(self.auto_trader.candidate_stocks)
        self.logger.info(
            f"스크리닝 결과: {len(self.auto_trader.candidate_stocks)}개 종목"
        )

    def test_market_risk_assessment(self):
        """시장 리스크 평가 테스트"""
        self.auto_trader._assess_market_risk()
        risk_status = self.auto_trader.risk_manager.risk_status
        self.assertIsNotNone(risk_status)
        self.logger.info(f"현재 리스크 상태: {risk_status}")

    def test_disclosure_check(self):
        """공시 확인 테스트"""
        self.auto_trader._check_disclosure()
        # 공시 정보는 있을 수도 있고 없을 수도 있음
        self.logger.info("공시 확인 완료")

    def test_morning_entry(self):
        """시초가 매매 테스트"""
        # 스크리닝 먼저 실행
        self.auto_trader._run_screening()
        # 시초가 매매 실행
        self.auto_trader._morning_entry()
        holdings = self.auto_trader.strategy.holdings
        self.logger.info(f"시초가 매매 결과: {len(holdings)}개 종목 보유")

    def test_position_monitoring(self):
        """포지션 모니터링 테스트"""
        # 스크리닝 및 진입 먼저 실행
        self.auto_trader._run_screening()
        self.auto_trader._morning_entry()
        # 포지션 모니터링 실행
        self.auto_trader._monitor_positions()
        self.logger.info("포지션 모니터링 완료")

    def test_portfolio_risk_monitoring(self):
        """포트폴리오 리스크 모니터링 테스트"""
        self.auto_trader._monitor_portfolio_risk()
        self.logger.info("포트폴리오 리스크 모니터링 완료")

    def test_news_monitoring(self):
        """뉴스 모니터링 테스트"""
        self.auto_trader._monitor_news()
        self.logger.info("뉴스 모니터링 완료")

    def test_closing_report(self):
        """클로징 리포트 생성 테스트"""
        report_file = self.auto_trader._generate_closing_report()
        self.assertIsNotNone(report_file)
        self.logger.info(f"클로징 리포트 생성 완료: {report_file}")

    def test_run_once(self):
        """1회 실행 테스트"""
        self.auto_trader.run_once()
        self.logger.info("1회 실행 완료")

    def test_backtest(self):
        """백테스트 실행 테스트"""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        self.auto_trader.run_backtest(start_date, end_date)
        self.logger.info(f"백테스트 완료 (기간: {start_date} ~ {end_date})")

    def test_full_cycle(self):
        """전체 사이클 테스트"""
        # 1. 초기화
        self.test_initialization()

        # 2. 장 전 준비
        self.test_market_risk_assessment()
        self.test_news_analysis()
        self.test_screening()
        self.test_disclosure_check()

        # 3. 장중 운영
        self.test_morning_entry()
        self.test_position_monitoring()
        self.test_portfolio_risk_monitoring()
        self.test_news_monitoring()

        # 4. 장 마감
        self.test_closing_report()

        self.logger.info("전체 사이클 테스트 완료")


if __name__ == "__main__":
    unittest.main()
