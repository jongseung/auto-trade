import logging
import time
import os
import schedule
from datetime import datetime, timedelta

import config
from api.kis_api import KisAPI
from screener.stock_screener import StockScreener
from strategy.trading_strategy import TradingStrategy
from risk.risk_manager import RiskManager
from backtesting.backtester import BackTester

logger = logging.getLogger("auto_trade.auto_trader")


class AutoTrader:
    """자동 매매 클래스"""

    def __init__(self, demo_mode=True):
        # 로깅 설정
        self.logger = config.setup_logging()

        # 데모 모드 설정
        self.demo_mode = demo_mode
        if self.demo_mode:
            logger.info("데모 모드로 실행합니다. 실제 API 호출은 이루어지지 않습니다.")

        # API 클라이언트 초기화
        self.api_client = KisAPI(demo_mode=self.demo_mode)

        # 스크리너 초기화
        self.screener = StockScreener(self.api_client)

        # 리스크 관리자 초기화
        self.risk_manager = RiskManager(self.api_client)

        # 매매 전략 초기화
        self.strategy = TradingStrategy(self.api_client, self.risk_manager)

        # 상태 변수
        self.running = False
        self.candidate_stocks = []

        # 스케줄러 초기화
        self.scheduler = schedule

        # 스케줄러 작업
        self.jobs = []

    def start(self):
        """자동 매매 시작"""
        logger.info("자동 매매를 시작합니다.")

        try:
            # 시장 리스크 평가
            try:
                self.risk_manager.assess_market_risk()
            except Exception as e:
                logger.error(f"시장 리스크 평가 중 오류 발생: {str(e)}")

            # 스케줄러 작업 등록
            self._register_jobs()

            # 스케줄러 시작 (무한 루프가 아닌 일회성 실행)
            self.running = True

            # 데모 모드 설정
            if self.demo_mode:
                logger.info(
                    "데모 모드로 실행 중입니다 - 실제 거래는 실행되지 않습니다."
                )

            # schedule 라이브러리는 run_pending()을 주기적으로 호출해야 함
            logger.info("스케줄러가 시작되었습니다. 등록된 작업 실행을 대기합니다.")

        except Exception as e:
            logger.error(f"자동 매매 시작 중 오류 발생: {str(e)}")

    def stop(self):
        """자동 매매 중지"""
        logger.info("자동 매매를 중지합니다.")

        try:
            # 스케줄러 중지
            self.running = False

            # 모든 예약 작업 취소
            schedule.clear()

            # 보유 종목 정보 업데이트
            self.strategy.update_holdings()

            # 보유 종목 있으면 청산
            if len(self.strategy.holdings) > 0:
                logger.info(f"보유 종목 청산: {len(self.strategy.holdings)}개")
                self.strategy.exit_all(reason="시스템 종료")
            else:
                logger.info("청산할 보유 종목이 없습니다.")
        except Exception as e:
            logger.error(f"자동 매매 중지 중 오류 발생: {str(e)}")

    def _register_jobs(self):
        """스케줄러 작업 등록"""
        # 시장 리스크 평가 (07:30)
        risk_assessment_job = (
            schedule.every().day.at("07:30:00").do(self._assess_market_risk)
        )
        self.jobs.append(risk_assessment_job)

        # 장 전 뉴스 분석 (07:30)
        morning_news_job = (
            schedule.every().day.at("07:30:00").do(self._run_news_analysis)
        )
        self.jobs.append(morning_news_job)

        # 장 전 스크리닝 (08:00)
        morning_screening_job = (
            schedule.every().day.at("08:00:00").do(self._run_screening)
        )
        self.jobs.append(morning_screening_job)

        # 프리장 모니터링 (08:30)
        pre_market_job = schedule.every().day.at("08:30:00").do(self._check_disclosure)
        self.jobs.append(pre_market_job)

        # 시초가 매매 (09:00)
        market_open_job = schedule.every().day.at("09:00:00").do(self._morning_entry)
        self.jobs.append(market_open_job)

        # 장중 뉴스 모니터링 (30분 단위)
        for hour in range(9, 15):
            for minute in [0, 30]:
                time_str = f"{hour:02d}:{minute:02d}:00"
                news_monitor_job = (
                    schedule.every().day.at(time_str).do(self._monitor_news)
                )
                self.jobs.append(news_monitor_job)

        # 장중 모니터링 (1분 단위)
        for hour in range(9, 15):
            for minute in range(0, 60, 1):
                time_str = f"{hour:02d}:{minute:02d}:00"
                market_monitor_job = (
                    schedule.every().day.at(time_str).do(self._monitor_positions)
                )
                self.jobs.append(market_monitor_job)

        # 리스크 모니터링 (30분 단위)
        for hour in range(9, 15):
            for minute in [0, 30]:
                time_str = f"{hour:02d}:{minute:02d}:00"
                risk_monitor_job = (
                    schedule.every().day.at(time_str).do(self._monitor_portfolio_risk)
                )
                self.jobs.append(risk_monitor_job)

        # 종가 클로징 리포트 (15:30)
        closing_report_job = (
            schedule.every().day.at("15:30:00").do(self._generate_closing_report)
        )
        self.jobs.append(closing_report_job)

        logger.info("스케줄러 작업 등록 완료")

    def _assess_market_risk(self):
        """시장 리스크 평가"""
        logger.info("시장 리스크 평가를 시작합니다.")
        self.risk_manager.assess_market_risk()

        # 리스크 상태에 따라 매매 전략 파라미터 조정
        risk_status = self.risk_manager.risk_status
        logger.info(f"현재 리스크 상태: {risk_status}")

        # 리스크 리포트 생성
        report_file = self.risk_manager.generate_risk_report()
        logger.info(f"리스크 리포트 생성 완료: {report_file}")

    def _run_news_analysis(self):
        """뉴스 분석 실행"""
        logger.info("뉴스 분석을 시작합니다.")

        try:
            # 뉴스 기반 스크리닝 실행
            news_candidates = self.screener.analyze_news_for_candidates(days=1)

            if news_candidates:
                # 뉴스 기반 종목 후보 설정
                self.news_candidates = news_candidates
                logger.info(f"뉴스 분석 완료: {len(news_candidates)}개 종목 추출")
            else:
                logger.info("뉴스 분석 완료: 후보 종목 없음")

        except Exception as e:
            logger.error(f"뉴스 분석 중 오류 발생: {str(e)}")

    def _run_screening(self):
        """종목 스크리닝 실행"""
        logger.info("종목 스크리닝을 시작합니다.")

        # 스크리닝 실행 (기술적 분석 + 뉴스 분석)
        self.candidate_stocks = self.screener.run_screening(include_news=True)

        # 매매 전략에 후보 종목 전달
        self.strategy.set_candidate_stocks(self.candidate_stocks)

        # 리스크 상태에 따른 최적 포트폴리오 계산
        if not self.demo_mode:
            account_info = self.api_client.get_account_info()
            if account_info:
                # 계좌 잔고 가져오기
                account_balance = float(account_info.get("총평가금액", 0))

                # 최적 포트폴리오 배분 계산
                optimal_allocation = (
                    self.risk_manager.calculate_optimal_portfolio_allocation(
                        self.candidate_stocks, account_balance
                    )
                )

                # 전략에 최적 포트폴리오 배분 전달
                self.strategy.set_portfolio_allocation(optimal_allocation)

        # 후보 종목 리포트 생성 및 저장
        report = self.screener.generate_report(self.candidate_stocks)
        self._save_report(report, "screening_report")

        logger.info(
            f"종목 스크리닝 완료: {len(self.candidate_stocks)}개 후보 종목 선별"
        )

    def _check_disclosure(self):
        """공시 정보 확인"""
        logger.info("공시 정보를 확인합니다.")

        disclosures = []
        for stock in self.candidate_stocks:
            ticker = stock["ticker"]
            stock_disclosures = self._check_stock_disclosure(ticker)
            disclosures.extend(stock_disclosures)

        if disclosures:
            report = self._generate_disclosure_report(disclosures)
            self._save_report(report, "disclosure_report")

            logger.info(f"공시 정보 확인 완료: {len(disclosures)}건 공시 발견")
        else:
            logger.info("공시 정보 확인 완료: 관련 공시 없음")

    def _check_stock_disclosure(self, ticker):
        """종목별 공시 정보 확인"""
        today = datetime.now().strftime("%Y%m%d")
        try:
            # get_disclosure는 리스트를 반환함
            disclosures = self.api_client.get_disclosure(today)

            # 리스트인지 확인하고 안전하게 처리
            if disclosures and isinstance(disclosures, list):
                # 해당 종목의 공시 필터링
                ticker_disclosures = [
                    disclosure
                    for disclosure in disclosures
                    if disclosure.get("mksc_shrn_iscd", "") == ticker
                ]
                return ticker_disclosures
            else:
                logger.warning(
                    f"공시 정보가 유효한 리스트 형식이 아닙니다: {type(disclosures)}"
                )
                return []
        except Exception as e:
            logger.error(f"공시 정보 처리 중 오류 발생: {str(e)}")
            return []

    def _generate_disclosure_report(self, disclosures):
        """공시 정보 리포트 생성"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = f"=== 공시 정보 리포트 ({now}) ===\n\n"

        try:
            for disclosure in disclosures:
                # 안전한 딕셔너리 접근을 위해 get 메서드 사용
                ticker = disclosure.get("mksc_shrn_iscd", "알 수 없음")
                stock_name = disclosure.get("corp_name", "알 수 없음")
                time = disclosure.get("dics_hora", "")
                title = disclosure.get("dics_tl", "")

                report += f"- {stock_name} ({ticker})\n"
                report += f"  시간: {time}\n"
                report += f"  제목: {title}\n\n"

        except Exception as e:
            logger.error(f"공시 정보 리포트 생성 중 오류 발생: {str(e)}")
            report += f"[오류 발생: {str(e)}]\n\n"

        report += "================================================================\n"
        return report

    def _morning_entry(self):
        """시초가 매매"""
        logger.info("시초가 매매를 시작합니다.")

        # 보유 종목 정보 업데이트
        self.strategy.update_holdings()

        # 최대 3개 종목만 진입 허용
        top_candidates = self.candidate_stocks[:3]

        for candidate in top_candidates:
            ticker = candidate["ticker"]

            # 진입 조건 체크
            ohlcv_data = self.api_client.get_ohlcv(ticker, period="D", count=5)

            if self.strategy.check_entry_condition(ticker, ohlcv_data):
                logger.info(f"{ticker} 진입 조건 충족 - 매수 실행")

                # 동적 포지션 사이즈 계산 (demo 모드가 아닌 경우)
                if not self.demo_mode:
                    account_info = self.api_client.get_account_info()
                    if account_info:
                        account_balance = float(account_info.get("총평가금액", 0))
                        current_price = ohlcv_data["close"].iloc[-1]

                        # 동적 포지션 사이즈 계산
                        position_size = self.risk_manager.calculate_position_size(
                            ticker, current_price, account_balance
                        )

                        # 동적 손절가 계산
                        stop_loss_price = self.risk_manager.calculate_dynamic_stoploss(
                            ticker, current_price
                        )

                        # 동적 익절가 계산
                        take_profit_price = (
                            self.risk_manager.calculate_dynamic_takeprofit(
                                ticker, current_price
                            )
                        )

                        # 매매 전략에 동적 매매 파라미터 전달
                        self.strategy.entry(
                            ticker,
                            amount=position_size,
                            stop_loss_price=stop_loss_price,
                            take_profit_price=take_profit_price,
                        )
                    else:
                        # 계좌 정보를 가져올 수 없는 경우 기본 설정으로 진입
                        self.strategy.entry(ticker)
                else:
                    # 데모 모드인 경우 기본 설정으로 진입
                    self.strategy.entry(ticker)

        logger.info("시초가 매매 완료")

    def _monitor_positions(self):
        """포지션 모니터링"""
        if len(self.strategy.holdings) == 0:
            return

        # 주문 체크 및 실행
        self.strategy.check_and_execute_orders()

    def _monitor_portfolio_risk(self):
        """포트폴리오 리스크 모니터링"""
        if self.demo_mode:
            return

        logger.info("포트폴리오 리스크 모니터링을 시작합니다.")

        try:
            # 계좌 정보 가져오기
            account_info = self.api_client.get_account_info()
            if not account_info:
                logger.warning("계좌 정보를 가져올 수 없습니다.")
                return

            # 포트폴리오 가치 계산
            portfolio_value = float(account_info.get("총평가금액", 0))

            # 일일 손실 한도 체크
            exceeds_daily_loss = self.risk_manager.track_daily_loss(portfolio_value)

            if exceeds_daily_loss:
                logger.warning("일일 손실 한도 초과: 모든 포지션 청산")
                self.strategy.exit_all(reason="일일 손실 한도 초과")

                # 리스크 리포트 생성
                self.risk_manager.generate_risk_report()

        except Exception as e:
            logger.error(f"포트폴리오 리스크 모니터링 중 오류 발생: {str(e)}")

    def _generate_closing_report(self):
        """클로징 리포트 생성"""
        logger.info("클로징 리포트를 생성합니다.")

        try:
            # 계좌 정보 조회
            account_info = self.api_client.get_account_info()
            if not account_info:
                logger.error("계좌 정보를 가져올 수 없습니다.")
                return

            # 보유 종목 정보 조회
            holdings = self.strategy.get_holdings_info()
            if not holdings:
                logger.warning("보유 종목 정보가 없습니다.")

            # 클로징 리포트 생성
            report = self._generate_account_report(account_info, holdings)
            self._save_report(report, "closing_report")

            logger.info("클로징 리포트 생성 완료")

        except Exception as e:
            logger.error(f"클로징 리포트 생성 중 오류 발생: {str(e)}")

    def _generate_account_report(self, account_info, holdings):
        """계좌 리포트 생성"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = f"=== 계좌 리포트 ({now}) ===\n\n"

        # 계좌 요약 정보
        report += "== 계좌 요약 ==\n"
        report += f"총평가금액: {account_info.get('total_value', '0')}원\n"
        report += f"매수금액: {account_info.get('balance', '0')}원\n"
        report += f"평가손익: {account_info.get('total_profit_loss', '0')}원\n"
        report += f"수익률: {account_info.get('total_profit_loss_rate', '0')}%\n"
        report += f"보유종목수: {len(account_info.get('positions', []))}개\n\n"

        # 보유 종목 정보
        report += "== 보유 종목 ==\n"
        report += "종목코드\t종목명\t수량\t매입가\t현재가\t평가손익\t수익률\n"

        for info in holdings:
            ticker = info.get("ticker", "")
            name = info.get("name", "")
            qty = info.get("quantity", 0)
            buy_price = info.get("average_price", 0)
            curr_price = info.get("current_price", 0)
            profit_loss = info.get("profit_loss", 0)
            return_rate = f"{info.get('profit_loss_rate', 0):.2f}"

            report += f"{ticker}\t{name}\t{qty}\t{buy_price}\t{curr_price}\t{profit_loss}\t{return_rate}%\n"

        report += "\n================================================================\n"
        return report

    def _save_report(self, report, report_type):
        """리포트 저장"""
        try:
            # 저장 디렉토리 설정
            report_dir = "data/reports"
            os.makedirs(report_dir, exist_ok=True)

            # 파일명 생성
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{report_dir}/{report_type}_{now}.txt"

            # 파일 저장
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(report)

            logger.info(f"리포트 저장 완료: {file_name}")
            return file_name

        except Exception as e:
            logger.error(f"리포트 저장 중 오류 발생: {str(e)}")
            return None

    def run_once(self):
        """1회 실행"""
        logger.info("1회 실행 모드를 시작합니다.")

        try:
            # 시장 리스크 평가
            self.risk_manager.assess_market_risk()

            # 뉴스 분석
            self._run_news_analysis()

            # 종목 스크리닝
            try:
                self._run_screening()
            except Exception as e:
                logger.error(f"종목 스크리닝 중 오류 발생: {str(e)}")
                # 스크리닝 실패 시 빈 리스트로 초기화하여 다음 단계 진행
                self.candidate_stocks = []

            # 공시 정보 확인 - 후보 종목이 있는 경우에만 실행
            if self.candidate_stocks:
                try:
                    self._check_disclosure()
                except Exception as e:
                    logger.error(f"공시 정보 확인 중 오류 발생: {str(e)}")
                    # 오류 발생해도 계속 진행
            else:
                logger.warning("후보 종목이 없어 공시 정보 확인을 건너뜁니다")

            # 초기 진입 - 후보 종목이 있는 경우에만 실행
            if self.candidate_stocks:
                try:
                    self._morning_entry()
                except Exception as e:
                    logger.error(f"초기 진입 중 오류 발생: {str(e)}")
            else:
                logger.warning("후보 종목이 없어 초기 진입을 건너뜁니다")

            # 클로징 리포트 생성
            try:
                self._generate_closing_report()
            except Exception as e:
                logger.error(f"클로징 리포트 생성 중 오류 발생: {str(e)}")

            logger.info("1회 실행 완료")

        except Exception as e:
            logger.error(f"1회 실행 중 오류 발생: {str(e)}")

    def run_backtest(self, start_date, end_date):
        """백테스팅 실행

        Args:
            start_date (str): 백테스팅 시작일 (YYYY-MM-DD)
            end_date (str): 백테스팅 종료일 (YYYY-MM-DD)
        """
        logger.info(f"백테스트 모드를 시작합니다. (기간: {start_date} ~ {end_date})")

        try:
            # 날짜 형식 변환 (YYYY-MM-DD -> YYYYMMDD)
            start_date_formatted = start_date.replace("-", "")
            end_date_formatted = end_date.replace("-", "")

            # 백테스터 초기화
            backtester = BackTester(
                self.api_client,
                start_date=start_date_formatted,
                end_date=end_date_formatted,
            )

            # 백테스팅 실행
            backtest_results = backtester.run_backtest()

            # 성과 시각화
            chart_file = backtester.plot_performance()

            # 결과 출력
            logger.info("===== 백테스팅 결과 =====")
            logger.info(f"초기 자본금: {backtest_results['initial_capital']:,.0f}원")
            logger.info(f"최종 자산가치: {backtest_results['final_nav']:,.0f}원")
            logger.info(f"총 수익률: {backtest_results['total_return_pct']:.2f}%")
            logger.info(f"연 수익률: {backtest_results['annual_return']:.2f}%")
            logger.info(f"승률: {backtest_results['win_rate']:.2f}%")
            logger.info(f"최대 낙폭(MDD): {backtest_results['mdd']:.2f}%")
            logger.info(f"샤프 비율: {backtest_results['sharpe_ratio']:.2f}")
            logger.info(f"성과 차트: {chart_file}")

            logger.info("백테스트 완료")

        except Exception as e:
            logger.error(f"백테스트 실행 중 오류 발생: {str(e)}")

    def _monitor_news(self):
        """장중 뉴스 모니터링"""
        logger.info("장중 뉴스 모니터링을 시작합니다.")
        try:
            # 최근 1시간 동안의 뉴스만 분석
            news_candidates = self.screener.analyze_news_for_candidates(
                days=0.04
            )  # 약 1시간

            if news_candidates:
                # 긴급 뉴스 필터링 (높은 점수나 긴급 키워드 포함)
                urgent_news = [
                    stock
                    for stock in news_candidates
                    if stock["score"] > 80 or "긴급" in stock.get("reason", "")
                ]

                if urgent_news:
                    logger.info(f"긴급 뉴스 발견: {len(urgent_news)}개")
                    # 긴급 뉴스에 대한 대응 로직
                    self._handle_urgent_news(urgent_news)

        except Exception as e:
            logger.error(f"장중 뉴스 모니터링 중 오류 발생: {str(e)}")

    def _handle_urgent_news(self, urgent_news):
        """긴급 뉴스 대응"""
        for news in urgent_news:
            ticker = news["ticker"]
            logger.info(f"긴급 뉴스 대응: {ticker} - {news['reason']}")

            # 보유 종목인 경우 청산 검토
            if ticker in self.strategy.holdings:
                should_exit, reason = self.strategy.check_exit_condition(ticker)
                if should_exit:
                    self.strategy.exit(ticker, reason=f"긴급 뉴스: {reason}")
