import logging
from datetime import datetime, time, timedelta
import pandas as pd
import time as time_module

import config
from api.kis_api import KisAPI
from utils.utils import is_trading_time, get_trading_time_status

logger = logging.getLogger("auto_trade.trading_strategy")


class TradingStrategy:
    """매매 전략 클래스"""

    def __init__(self, api_client, risk_manager, candidate_stocks=None):
        """
        Args:
            api_client (KisAPI): 한국투자증권 API 클라이언트
            risk_manager (RiskManager): 리스크 관리 클래스
            candidate_stocks (list): 후보 종목 리스트
        """
        self.api_client = api_client
        self.risk_manager = risk_manager
        self.candidate_stocks = candidate_stocks or []
        self.holdings = {}  # 보유 종목 정보
        self.order_history = []  # 주문 이력
        self.pending_orders = {}  # 미체결 주문 추적

        # 전략 설정
        self.take_profit_ratio = config.TAKE_PROFIT_RATIO
        self.stop_loss_ratio = config.STOP_LOSS_RATIO
        self.trailing_stop = config.TRAILING_STOP
        self.max_stocks = config.MAX_STOCKS
        self.portfolio_ratio = config.PORTFOLIO_RATIO
        self.stock_ratio = config.STOCK_RATIO
        self.max_daily_loss = config.MAX_DAILY_LOSS

        # 시간 설정
        self.morning_entry_start = self._str_to_time(config.MORNING_ENTRY_START)
        self.morning_entry_end = self._str_to_time(config.MORNING_ENTRY_END)
        self.additional_entry_end = self._str_to_time(config.ADDITIONAL_ENTRY_END)
        self.market_close = self._str_to_time(config.MARKET_CLOSE)

        # 수수료 및 슬리피지 설정
        self.commission_rate = 0.00015  # 수수료 0.015%
        self.tax_rate = 0.003  # 거래세 0.3% (매도 시에만 적용)
        self.avg_slippage_rate = 0.001  # 평균 슬리피지 0.1%
        self.order_retry_count = 3  # 주문 재시도 횟수
        self.partial_fill_wait_time = 30  # 부분 체결 대기 시간(초)

    def _str_to_time(self, time_str):
        """시간 문자열을 time 객체로 변환"""
        hours, minutes = map(int, time_str.split(":"))
        return time(hours, minutes)

    def set_candidate_stocks(self, candidate_stocks):
        """후보 종목 설정

        Args:
            candidate_stocks (list): 후보 종목 리스트
        """
        self.candidate_stocks = candidate_stocks

    def set_portfolio_allocation(self, allocation):
        """포트폴리오 배분 설정

        Args:
            allocation (dict): 종목별 최적 배분 비율
        """
        self.portfolio_allocation = allocation
        logger.info(f"포트폴리오 배분 설정 완료: {len(allocation)}개 종목")

        # 배분 정보 로깅
        if allocation:
            for ticker, info in allocation.items():
                logger.info(
                    f"포트폴리오 배분: {ticker} ({info.get('name', '')}): "
                    f"{info.get('weight', 0):.2%}, 목표금액: {info.get('target_amount', 0):,.0f}원"
                )

    def update_holdings(self):
        """보유 종목 정보 업데이트"""
        account_info = self.api_client.get_account_info()
        if not account_info:
            logger.warning("계좌 정보를 가져올 수 없습니다.")
            return

        # 현재 보유 종목 정보 가져오기
        positions = account_info.get("positions", [])

        # 기존 보유 정보 백업 (고점 등의 정보 유지를 위해)
        holdings_backup = self.holdings.copy()

        # 홀딩 정보 초기화
        self.holdings = {}

        # 업데이트된 보유 종목 정보 추가
        for position in positions:
            ticker = position.get("ticker", "")
            if not ticker:
                continue

            # 이미 백업에 있는 정보 가져와서 업데이트
            current_info = {}
            if ticker in holdings_backup:
                current_info = holdings_backup[ticker].copy()

            # 현재 정보로 업데이트
            current_info["ticker"] = ticker
            current_info["name"] = position.get("name", "")
            current_info["quantity"] = position.get("quantity", 0)
            current_info["buy_price"] = position.get("buy_price", 0)
            current_info["current_price"] = position.get("current_price", 0)
            current_info["profit_loss"] = position.get("eval_profit_loss", 0)
            current_info["eval_amount"] = position.get("eval_amount", 0)

            # 이전 정보가 없으면 초기화
            if "entry_time" not in current_info:
                current_info["entry_time"] = datetime.now()
            if "high_price" not in current_info:
                current_info["high_price"] = position.get("current_price", 0)
            elif position.get("current_price", 0) > current_info["high_price"]:
                current_info["high_price"] = position.get("current_price", 0)

            # 홀딩에 추가
            self.holdings[ticker] = current_info

        logger.info(f"보유 종목 정보 업데이트 완료: {len(self.holdings)}개 종목")

    def calculate_buy_amount(self, ticker, price):
        """매수 금액 계산 (수수료와 슬리피지 고려)

        Args:
            ticker (str): 종목코드
            price (int): 종목 가격

        Returns:
            tuple: (매수 수량, 매수 금액)
        """
        # 계좌 정보 조회
        account_info = self.api_client.get_account_info()

        if account_info is None:
            logger.error("계좌 정보를 가져오는데 실패했습니다.")
            return 0, 0

        # 총 평가금액
        total_eval_amount = account_info["total_evaluated_amount"]

        # 포트폴리오 금액 (총 자산의 N%)
        portfolio_amount = total_eval_amount * self.portfolio_ratio

        # 종목별 최대 투자 금액 (총 자산의 N%)
        max_stock_amount = total_eval_amount * self.stock_ratio

        # 현재 보유 종목 개수
        current_holdings = len(self.holdings)

        # 매수 가능 종목 수
        available_stocks = self.max_stocks - current_holdings

        if available_stocks <= 0:
            logger.warning(f"최대 보유 종목 수({self.max_stocks}개)에 도달했습니다.")
            return 0, 0

        # 예상 슬리피지 계산 (현재가 기준)
        expected_slippage = price * self.avg_slippage_rate

        # 수수료 포함 예상 매수가
        expected_buy_price = price + expected_slippage

        # 매수 시 수수료 고려
        commission_adjusted_price = expected_buy_price * (1 + self.commission_rate)

        # 매수 금액 계산 (수수료와 슬리피지 고려)
        base_buy_amount = min(portfolio_amount / available_stocks, max_stock_amount)

        # 수수료와 슬리피지를 고려한 실제 투자 가능 금액
        adjusted_buy_amount = base_buy_amount / (
            1 + self.commission_rate + self.avg_slippage_rate
        )

        # 매수 수량 계산 (1주 단위 내림)
        quantity = int(adjusted_buy_amount / price)

        # 최소 1주 이상
        if quantity <= 0:
            quantity = 1

        # 최종 매수 금액 (예상)
        final_amount = price * quantity

        logger.info(
            f"매수 금액 계산: {ticker} - {quantity}주 ({final_amount:,}원) [수수료+슬리피지 고려]"
        )
        return quantity, final_amount

    def check_entry_condition(self, ticker, ohlcv_data=None):
        """진입 조건 체크

        Args:
            ticker (str): 종목코드
            ohlcv_data (DataFrame): OHLCV 데이터

        Returns:
            bool: 진입 조건 충족 여부
        """
        # 거래 시간 체크
        trading_status = get_trading_time_status()
        if trading_status not in ["REGULAR", "OPENING_AUCTION"]:
            return False

        now = datetime.now().time()

        # 시간 체크 (09:00~09:05)
        if not (self.morning_entry_start <= now <= self.morning_entry_end):
            return False

        # 이미 보유 중인 종목이면 추가 매수 여부 확인
        if ticker in self.holdings:
            # 추가 진입 시간이 지났으면 추가 매수 안함
            if now > self.additional_entry_end:
                return False

            # TODO: 추가 매수 로직 구현 (모멘텀, 2차 갭업, 거래량 재급증 등)
            return False

        # 후보 종목에 없으면 매수 안함
        if not any(stock["ticker"] == ticker for stock in self.candidate_stocks):
            return False

        # 보유 종목 수 체크
        if len(self.holdings) >= self.max_stocks:
            logger.warning(f"최대 보유 종목 수({self.max_stocks}개)에 도달했습니다.")
            return False

        # OHLCV 데이터가 없으면 가져오기
        if ohlcv_data is None:
            ohlcv_data = self.api_client.get_ohlcv(ticker, period="D", count=5)

            if ohlcv_data.empty:
                logger.error(f"{ticker} OHLCV 데이터를 가져오는데 실패했습니다.")
                return False

        # 상승 모멘텀 체크 (최근 N일 연속 상승)
        if len(ohlcv_data) >= 3:
            prev_closes = ohlcv_data["close"].values[-3:]

            # 최근 3일 연속 상승 여부
            if prev_closes[0] < prev_closes[1] < prev_closes[2]:
                # 거래량 증가 여부 (전일 대비 1.5배 이상)
                if len(ohlcv_data) >= 2:
                    prev_volumes = ohlcv_data["volume"].values[-2:]
                    if prev_volumes[1] > prev_volumes[0] * 1.5:
                        logger.info(
                            f"{ticker} 모멘텀 확인: 3일 연속 상승 + 거래량 증가"
                        )
                        return True

        return False

    def check_exit_condition(self, ticker):
        """청산 조건 체크

        Args:
            ticker (str): 종목코드

        Returns:
            tuple: (청산 여부, 청산 사유)
        """
        # 거래 시간 체크 - 장 외에도 청산은 가능하도록 유지 (단, 정규장 시간이 아닌 경우 경고 로그)
        trading_status = get_trading_time_status()
        if trading_status not in ["REGULAR", "CLOSING_AUCTION"]:
            logger.warning(
                f"정규장 시간이 아닙니다: {trading_status}. 긴급 청산 조건만 확인합니다."
            )

        if ticker not in self.holdings:
            return False, "보유 종목 아님"

        # 종목 정보 가져오기
        holding_info = self.holdings[ticker]

        # 현재가 업데이트
        current_price = self.api_client.get_current_price(ticker)

        if current_price is None:
            logger.error(f"{ticker} 현재가를 가져오는데 실패했습니다.")
            return False, "현재가 조회 실패"

        # 보유 정보 업데이트
        holding_info["current_price"] = current_price
        holding_info["profit_loss"] = (
            current_price / holding_info["buy_price"] - 1
        ) * 100
        holding_info["high_price"] = max(holding_info["high_price"], current_price)

        # 1. 익절 조건 (목표 수익률 도달)
        profit_ratio = current_price / holding_info["buy_price"] - 1
        if profit_ratio >= self.take_profit_ratio:
            return True, f"익절 조건 충족: {profit_ratio:.2%}"

        # 2. 손절 조건 (목표 손실률 도달)
        if profit_ratio <= -self.stop_loss_ratio:
            return True, f"손절 조건 충족: {profit_ratio:.2%}"

        # 3. 트레일링 스탑 조건
        trailing_ratio = 1 - (current_price / holding_info["high_price"])
        if (
            trailing_ratio >= self.trailing_stop
            and holding_info["high_price"] > holding_info["buy_price"]
        ):
            return True, f"트레일링 스탑 발동: 고점 대비 {trailing_ratio:.2%} 하락"

        # 4. 장 마감 임박 조건
        if trading_status == "CLOSING_AUCTION":
            return True, "장 마감 임박"

        # 5. 보유 기간 초과 (최대 3일)
        entry_time = holding_info["entry_time"]
        holding_days = (datetime.now() - entry_time).days
        if holding_days >= 3:
            return True, f"최대 보유 기간 초과: {holding_days}일"

        return False, "청산 조건 미충족"

    def calculate_daily_profit_loss(self):
        """일일 손익률 계산

        Returns:
            float: 일일 손익률 (%)
        """
        if not self.holdings:
            return 0.0

        total_buy_amount = sum(
            holding["buy_price"] * holding["quantity"]
            for holding in self.holdings.values()
        )

        total_eval_amount = sum(
            holding["current_price"] * holding["quantity"]
            for holding in self.holdings.values()
        )

        # 매입금액이 0인 경우 0 반환
        if total_buy_amount == 0:
            return 0.0

        profit_loss_ratio = (total_eval_amount / total_buy_amount - 1) * 100
        return profit_loss_ratio

    def check_daily_loss_limit(self):
        """일일 손실 한도 체크

        Returns:
            bool: 손실 한도 초과 여부
        """
        daily_profit_loss = self.calculate_daily_profit_loss()

        if daily_profit_loss <= -self.max_daily_loss * 100:
            logger.warning(f"일일 손실 한도 초과: {daily_profit_loss:.2f}%")
            return True

        return False

    def entry(self, ticker, quantity=0, price=0):
        """종목 진입 (매수)

        Args:
            ticker (str): 종목코드
            quantity (int): 매수 수량 (0일 경우 자동 계산)
            price (int): 매수 가격 (0일 경우 시장가)

        Returns:
            dict: 매수 결과
        """
        # 거래 시간 체크
        trading_status = get_trading_time_status()
        if trading_status not in ["REGULAR", "OPENING_AUCTION"]:
            logger.warning(f"거래 불가능 시간: {trading_status}")
            return None

        # 계좌 정보 가져오기
        account_info = self.api_client.get_account_info()
        if not account_info:
            logger.warning("계좌 정보를 가져올 수 없습니다.")
            return None

        # 현재가 조회
        current_price = price
        if current_price <= 0:
            current_price = self.api_client.get_current_price(ticker)
            if not current_price:
                logger.error(f"{ticker} 현재가를 가져오는데 실패했습니다.")
                return None

        # 매수 수량 결정
        if quantity <= 0:
            available_amount = account_info.get("available_amount", 0)
            total_evaluated_amount = account_info.get("total_evaluated_amount", 0)

            # 리스크 매니저를 통한 동적 포지션 크기 계산
            position_size = self.risk_manager.calculate_position_size(
                ticker, current_price, total_evaluated_amount
            )

            # 수량 계산
            quantity, _ = self.calculate_buy_quantity(
                ticker, current_price, position_size
            )

        # 주문 가능 여부 확인 (컴플라이언스 체크)
        can_order, reason = self.risk_manager.can_trade(
            ticker, "buy", quantity, current_price
        )
        if not can_order:
            logger.warning(f"{ticker} 매수 불가: {reason}")
            return None

        # 매수 주문 실행
        logger.info(f"매수 주문: {ticker} {quantity}주 @ {current_price:,}원")

        # 시장가 매수 주문
        order_result = self.api_client.market_buy(ticker, quantity)
        if not order_result:
            logger.error(f"{ticker} 매수 주문 실패")
            return None

        # 주문 기록
        self.risk_manager.record_order(ticker, "buy")

        # 주문 정보 기록
        order_info = {
            "ticker": ticker,
            "order_no": order_result.get("order_no", ""),
            "order_id": order_result.get("order_id", ""),
            "order_time": datetime.now(),
            "last_check_time": datetime.now(),
            "quantity": quantity,
            "price": current_price,
            "order_type": "buy",
            "status": "pending",
            "retry_count": 0,
        }

        # 주문 이력에 추가
        self.order_history.append(
            {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ticker": ticker,
                "action": "buy",
                "quantity": quantity,
                "price": current_price,
                "order_id": order_result.get("order_id", ""),
                "status": "submitted",
            }
        )

        # 미체결 주문 추적
        self.pending_orders[order_result.get("order_id", "")] = order_info

        return order_result

    def calculate_buy_quantity(self, ticker, price, buy_amount):
        """매수 수량 계산

        Args:
            ticker (str): 종목코드
            price (int): 주가
            buy_amount (int): 매수 금액

        Returns:
            tuple: (매수 수량, 최종 매수 금액)
        """
        if price <= 0:
            logger.error(f"가격이 0 이하입니다: {ticker}")
            return 0, 0

        if buy_amount <= 0:
            logger.error(f"매수 금액이 0 이하입니다: {ticker}")
            return 0, 0

        # 수수료와 슬리피지를 고려한 실제 투자 가능 금액
        adjusted_buy_amount = buy_amount / (
            1 + self.commission_rate + self.avg_slippage_rate
        )

        # 매수 수량 계산 (1주 단위 내림)
        quantity = int(adjusted_buy_amount / price)

        # 최소 1주 이상
        if quantity <= 0:
            quantity = 1

        # 최종 매수 금액 (예상)
        final_amount = price * quantity

        logger.info(
            f"매수 금액 계산: {ticker} - {quantity}주 ({final_amount:,}원) [수수료+슬리피지 고려]"
        )
        return quantity, final_amount

    def exit_all(self, reason=""):
        """전체 포지션 청산

        Args:
            reason (str): 청산 사유
        """
        if not self.holdings:
            logger.info("청산할 보유 종목이 없습니다.")
            return

        logger.warning(f"전체 포지션 청산 실행: {reason}")

        for ticker in list(self.holdings.keys()):
            self.exit(ticker, reason=reason)

    def exit(
        self,
        ticker,
        quantity=None,
        reason="",
        use_limit_order=False,
        price_adjust_pct=0.005,
    ):
        """종목 매도

        Args:
            ticker (str): 종목코드
            quantity (int): 매도 수량 (None일 경우 전량 매도)
            reason (str): 매도 사유
            use_limit_order (bool): 지정가 주문 사용 여부
            price_adjust_pct (float): 지정가 주문 시 호가 조정 비율 (기본 0.5%)

        Returns:
            dict: 매도 결과
        """
        # 거래 시간 체크 - 청산은 장 외에도 가능하도록 설정
        trading_status = get_trading_time_status()
        if trading_status == "CLOSED":
            logger.warning("장이 완전히 종료되어 매도가 불가능합니다.")
            return None

        if ticker not in self.holdings:
            logger.warning(f"{ticker}은(는) 보유 종목이 아닙니다.")
            return None

        # 매도 수량 결정
        if quantity is None or quantity <= 0:
            quantity = self.holdings[ticker]["quantity"]

        current_price = self.holdings[ticker]["current_price"]

        # 주문 가능 여부 확인 (컴플라이언스 체크)
        can_order, reason_text = self.risk_manager.can_trade(
            ticker, "sell", quantity, current_price
        )
        if not can_order:
            logger.warning(
                f"{ticker} 매도 불가: {reason_text}. 매도가 긴급히 필요한 경우 재시도합니다."
            )
            # 매도는 긴급 상황에서도 가능해야 하므로 경고만 기록

        # 주문 방식 선택
        if use_limit_order:
            # 지정가 매도 주문 (현재가 - 0.5% 정도로 호가 설정)
            limit_price = int(current_price * (1 - price_adjust_pct))
            order_result = self.api_client.limit_sell(ticker, quantity, limit_price)
            order_type = "limit_sell"
            logger.info(
                f"지정가 매도 주문: {ticker} {quantity}주 @ {limit_price:,}원 (현재가 {current_price:,}원) [{reason}]"
            )
        else:
            # 시장가 매도 주문
            order_result = self.api_client.market_sell(ticker, quantity)
            order_type = "market_sell"
            logger.info(
                f"시장가 매도 주문: {ticker} {quantity}주 @ {current_price:,}원 [{reason}]"
            )

        if order_result is None:
            logger.error(f"{ticker} 매도 주문 실패")
            return None

        # 주문 기록
        self.risk_manager.record_order(ticker, "sell")

        # 주문 정보 기록
        order_info = {
            "ticker": ticker,
            "order_no": order_result.get("order_no", ""),
            "order_id": order_result.get("order_id", ""),
            "order_time": datetime.now(),
            "last_check_time": datetime.now(),
            "quantity": quantity,
            "price": current_price if order_type == "market_sell" else limit_price,
            "order_type": order_type,
            "status": "pending",
            "retry_count": 0,
            "reason": reason,
        }

        # 주문 이력에 추가
        self.order_history.append(
            {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ticker": ticker,
                "action": "sell",
                "quantity": quantity,
                "price": current_price if order_type == "market_sell" else limit_price,
                "order_id": order_result.get("order_id", ""),
                "status": "submitted",
                "reason": reason,
            }
        )

        # 미체결 주문 추적
        self.pending_orders[order_result.get("order_id", "")] = order_info

        return order_result

    def check_and_execute_orders(self):
        """주문 체크 및 실행"""
        # 거래 시간 체크
        trading_status = get_trading_time_status()
        if trading_status not in ["REGULAR", "OPENING_AUCTION", "CLOSING_AUCTION"]:
            logger.debug(f"주문 실행 불가능 시간: {trading_status}")

            # 장 마감 후에는 미체결 주문만 취소
            if trading_status == "POST_MARKET" and self.pending_orders:
                logger.info("장 마감 후 미체결 주문 취소")
                for order_id, order_info in list(self.pending_orders.items()):
                    self._cancel_pending_order(order_id, order_info)

            return

        # 1. 보유 종목 정보 업데이트
        self.update_holdings()

        # 1-1. 미체결 주문 확인 및 처리
        pending_results = self.check_pending_orders()
        if pending_results and sum(pending_results.values()) > 0:
            logger.info(
                f"미체결 주문 처리: 체결완료 {pending_results['filled']}건, 부분체결 {pending_results['partial']}건, 재시도 {pending_results['retry']}건, 취소 {pending_results['cancelled']}건"
            )

        # 2. 일일 손실 한도 체크
        if self.check_daily_loss_limit():
            logger.warning("일일 손실 한도 초과로 인한 전량 청산")
            self.exit_all(reason="일일 손실 한도 초과")
            return

        # 3. 청산 조건 체크
        for ticker in list(self.holdings.keys()):
            should_exit, reason = self.check_exit_condition(ticker)
            if should_exit:
                logger.info(f"{ticker} 청산 조건 충족: {reason}")
                self.exit(ticker, reason=reason)

        # 4. 진입 조건 체크 (후보 종목 중 상위 3개만)
        # 정규장 시간에만 새로운 진입 허용
        if trading_status not in ["REGULAR", "OPENING_AUCTION"]:
            logger.debug(f"진입 불가능 시간: {trading_status}")
            return

        now = datetime.now().time()
        top_candidates = self.candidate_stocks[:3]

        # 시초가 매매 시간에만 새로운 진입 허용
        if self.morning_entry_start <= now <= self.morning_entry_end:
            for candidate in top_candidates:
                ticker = candidate["ticker"]

                if ticker not in self.holdings and len(self.holdings) < self.max_stocks:
                    ohlcv_data = self.api_client.get_ohlcv(ticker, period="D", count=5)

                    if self.check_entry_condition(ticker, ohlcv_data):
                        logger.info(f"{ticker} 진입 조건 충족")
                        self.entry(ticker)

    def _cancel_pending_order(self, order_id, order_info):
        """미체결 주문 취소

        Args:
            order_id (str): 주문 ID
            order_info (dict): 주문 정보
        """
        logger.info(f"미체결 주문 취소: {order_info['ticker']} ({order_id})")

        # 기존 주문 취소
        cancel_result = self.api_client.cancel_order(
            order_info.get("order_no", ""),
            order_info["ticker"],
            order_info["quantity"],
            "00" if "limit" in order_info["order_type"] else "01",
        )

        if cancel_result and cancel_result.get("success"):
            logger.info(f"주문 취소 성공: {order_info['ticker']} ({order_id})")

            # 주문 이력 업데이트
            self.order_history.append(
                {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ticker": order_info["ticker"],
                    "action": "cancel",
                    "quantity": order_info["quantity"],
                    "price": order_info["price"],
                    "order_id": order_id,
                    "status": "cancelled",
                    "reason": "미체결 주문 취소",
                }
            )

            # 미체결 주문 목록에서 제거
            if order_id in self.pending_orders:
                del self.pending_orders[order_id]

            return True
        else:
            logger.warning(f"주문 취소 실패: {order_info['ticker']} ({order_id})")
            return False

    def check_pending_orders(self):
        """미체결 주문 확인 및 처리

        미체결 주문에 대해 체결 상태를 확인하고, 일정 시간 이상 미체결 상태인 경우
        재시도 또는 취소 처리를 수행합니다.

        Returns:
            dict: 처리된 주문 현황
        """
        if not self.pending_orders:
            return {"filled": 0, "partial": 0, "cancelled": 0, "retry": 0}

        result_stats = {"filled": 0, "partial": 0, "cancelled": 0, "retry": 0}
        orders_to_remove = []

        # 현재 시간
        now = datetime.now()

        # 거래 시간 체크
        trading_status = get_trading_time_status()

        for order_id, order_info in self.pending_orders.items():
            # 주문 상태 확인
            order_status = self.api_client.get_order_status(
                order_info.get("order_no", "")
            )

            if order_status is None:
                logger.warning(f"주문 상태 조회 실패: {order_id}")
                continue

            # 체결 완료
            if order_status["remained_quantity"] == 0:
                logger.info(f"주문 체결 완료: {order_info['ticker']} ({order_id})")
                orders_to_remove.append(order_id)
                result_stats["filled"] += 1
                continue

            # 부분 체결 상태인 경우
            if 0 < order_status["remained_quantity"] < order_status["order_quantity"]:
                # 부분 체결 후 대기 시간 체크
                elapsed_time = (now - order_info["last_check_time"]).total_seconds()

                if elapsed_time >= self.partial_fill_wait_time:
                    # 남은 수량에 대해 추가 조치 (취소 또는 재시도)
                    logger.info(
                        f"부분 체결 후 대기시간 초과: {order_info['ticker']} ({order_id})"
                    )

                    # 기존 주문 취소
                    cancel_result = self.api_client.cancel_order(
                        order_info.get("order_no", ""),
                        order_info["ticker"],
                        order_status["remained_quantity"],
                        "00" if "limit" in order_info["order_type"] else "01",
                    )

                    if cancel_result and cancel_result["success"]:
                        logger.info(f"부분 체결 주문 취소 성공: {order_info['ticker']}")
                    else:
                        logger.warning(
                            f"부분 체결 주문 취소 실패: {order_info['ticker']}"
                        )

                    # 남은 수량에 대해 시장가로 재시도
                    remained_qty = order_status["remained_quantity"]

                    # 거래 시간이 아니면 재시도 안함
                    if trading_status not in [
                        "REGULAR",
                        "OPENING_AUCTION",
                        "CLOSING_AUCTION",
                    ]:
                        logger.warning(
                            f"거래 시간이 아니므로 남은 수량({remained_qty}주)에 대한 재시도를 하지 않습니다."
                        )
                    else:
                        if order_info["order_type"] == "buy":
                            # 매수 주문 재시도
                            new_order = self.entry(
                                order_info["ticker"], quantity=remained_qty
                            )
                        else:
                            # 매도 주문 재시도
                            new_order = self.exit(
                                order_info["ticker"],
                                quantity=remained_qty,
                                reason=order_info.get("reason", "부분체결 후 재시도"),
                            )

                        if new_order:
                            logger.info(
                                f"남은 수량({remained_qty}주)에 대한 주문 재시도: {order_info['ticker']}"
                            )
                            result_stats["retry"] += 1

                    orders_to_remove.append(order_id)
                    result_stats["partial"] += 1
                else:
                    # 대기 시간이 지나지 않았으면 계속 대기
                    logger.info(
                        f"부분 체결 상태: {order_info['ticker']} ({order_id}), 체결: {order_status['executed_quantity']}주, 잔량: {order_status['remained_quantity']}주"
                    )
                    order_info["last_check_time"] = now

            # 전량 미체결 상태인 경우
            elif order_status["remained_quantity"] == order_status["order_quantity"]:
                # 주문 시간으로부터 경과 시간 계산
                elapsed_time = (now - order_info["order_time"]).total_seconds()

                # 일정 시간(2분) 이상 미체결 상태인 경우
                if elapsed_time > 120:
                    # 거래 시간이 아니면 취소만 하고 재시도 안함
                    if trading_status not in [
                        "REGULAR",
                        "OPENING_AUCTION",
                        "CLOSING_AUCTION",
                    ]:
                        logger.warning(
                            f"거래 시간이 아니므로 미체결 주문 취소만 수행합니다: {order_info['ticker']}"
                        )
                        self._cancel_pending_order(order_id, order_info)
                        orders_to_remove.append(order_id)
                        result_stats["cancelled"] += 1
                        continue

                    # 재시도 횟수 확인
                    if order_info["retry_count"] < self.order_retry_count:
                        # 기존 주문 취소
                        cancel_result = self.api_client.cancel_order(
                            order_info.get("order_no", ""),
                            order_info["ticker"],
                            order_status["order_quantity"],
                            "00" if "limit" in order_info["order_type"] else "01",
                        )

                        if not cancel_result or not cancel_result["success"]:
                            logger.warning(
                                f"미체결 주문 취소 실패: {order_info['ticker']}"
                            )
                            continue

                        # 시장가로 재시도
                        if order_info["order_type"] == "buy":
                            # 매수 주문 재시도
                            new_order = self.entry(
                                order_info["ticker"],
                                quantity=order_status["order_quantity"],
                            )
                        else:
                            # 매도 주문 재시도
                            new_order = self.exit(
                                order_info["ticker"],
                                quantity=order_status["order_quantity"],
                                reason=order_info.get("reason", "미체결 주문 재시도"),
                            )

                        if new_order:
                            logger.info(f"미체결 주문 재시도: {order_info['ticker']}")
                            # 새 주문의 재시도 횟수 업데이트
                            order_id_new = new_order.get("order_id", "")
                            if order_id_new in self.pending_orders:
                                self.pending_orders[order_id_new]["retry_count"] = (
                                    order_info["retry_count"] + 1
                                )

                            result_stats["retry"] += 1
                        else:
                            logger.warning(
                                f"미체결 주문 재시도 실패: {order_info['ticker']}"
                            )
                            result_stats["cancelled"] += 1

                        orders_to_remove.append(order_id)
                    else:
                        # 재시도 횟수 초과 - 취소만 처리
                        logger.warning(
                            f"재시도 횟수 초과: {order_info['ticker']} ({order_info['retry_count']}/{self.order_retry_count})"
                        )
                        cancel_result = self.api_client.cancel_order(
                            order_info.get("order_no", ""),
                            order_info["ticker"],
                            order_status["order_quantity"],
                            "00" if "limit" in order_info["order_type"] else "01",
                        )

                        if cancel_result and cancel_result["success"]:
                            logger.info(f"주문 취소 성공: {order_info['ticker']}")
                            result_stats["cancelled"] += 1
                        else:
                            logger.warning(f"주문 취소 실패: {order_info['ticker']}")

                        orders_to_remove.append(order_id)

        # 처리 완료된 주문 제거
        for order_id in orders_to_remove:
            if order_id in self.pending_orders:
                del self.pending_orders[order_id]

        return result_stats

    def generate_performance_report(self):
        """성과 리포트 생성

        Returns:
            str: 성과 리포트 문자열
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = f"=== 매매 성과 리포트 ({now}) ===\n\n"

        # 일일 손익 계산
        daily_profit_loss = self.calculate_daily_profit_loss()
        report += f"일일 손익률: {daily_profit_loss:.2f}%\n\n"

        # 보유 종목 정보
        report += f"보유 종목 ({len(self.holdings)}/{self.max_stocks}):\n"
        for ticker, info in self.holdings.items():
            report += f"- {info['name']} ({ticker})\n"
            report += f"  수량: {info['quantity']}주\n"
            report += f"  매입가: {info['buy_price']:,}원\n"
            report += f"  현재가: {info['current_price']:,}원\n"
            report += f"  손익률: {info['profit_loss']:.2f}%\n"
            report += f"  평가금액: {info['eval_amount']:,}원\n\n"

        # 당일 주문 이력
        today = datetime.now().date()
        today_orders = [
            order for order in self.order_history if order["order_time"].date() == today
        ]

        if today_orders:
            report += f"당일 주문 이력 ({len(today_orders)}건):\n"
            for order in today_orders:
                order_type = "매수" if order["order_type"] == "buy" else "매도"
                report += f"- {order_type}: {order['ticker']} {order['quantity']}주 @ {order['price']:,}원\n"
                report += f"  시간: {order['order_time'].strftime('%H:%M:%S')}\n"
                if "reason" in order and order["reason"]:
                    report += f"  사유: {order['reason']}\n"
                report += f"  상태: {order['status']}\n\n"

        report += "================================================================\n"
        return report

    def get_holdings_info(self):
        """보유 종목 정보 반환

        Returns:
            list: 보유 종목 정보 리스트
        """
        # 최신 정보로 업데이트
        self.update_holdings()

        holdings_list = []

        for ticker, data in self.holdings.items():
            # 수익률 계산
            if data.get("buy_price") and data.get("current_price"):
                profit_pct = (data["current_price"] / data["buy_price"] - 1) * 100
            else:
                profit_pct = 0

            # 홀딩 정보 구성
            holding_info = {
                "ticker": ticker,
                "name": data.get("name", ""),
                "quantity": data.get("quantity", 0),
                "buy_price": data.get("buy_price", 0),
                "current_price": data.get("current_price", 0),
                "profit_loss": data.get("profit_loss", 0),
                "profit_pct": profit_pct,
                "eval_amount": data.get("eval_amount", 0),
                "entry_time": data.get("entry_time", datetime.now()),
                "high_price": data.get("high_price", 0),
                "holding_days": (
                    datetime.now() - data.get("entry_time", datetime.now())
                ).days,
            }

            holdings_list.append(holding_info)

        return holdings_list

    def rebalance_portfolio(self, target_allocation):
        """포트폴리오 리밸런싱 실행

        Args:
            target_allocation (dict): 목표 자산 배분 비율
        """
        logger.info("포트폴리오 리밸런싱 시작")

        # 현재 포트폴리오 상태 업데이트
        self.update_holdings()

        # 계좌 정보 조회
        account_info = self.api_client.get_account_info()
        if not account_info:
            logger.error("계좌 정보를 가져오는데 실패했습니다.")
            return

        total_value = account_info["total_evaluated_amount"]

        # 현재 보유 종목별 목표 금액 계산
        target_amounts = {}
        for ticker, ratio in target_allocation.items():
            target_amounts[ticker] = total_value * ratio

        # 리밸런싱 필요한 종목 식별
        rebalance_orders = []
        for ticker, target_amount in target_amounts.items():
            current_amount = self.holdings.get(ticker, {}).get("eval_amount", 0)
            diff_amount = target_amount - current_amount

            if (
                abs(diff_amount) > total_value * 0.01
            ):  # 1% 이상 차이나는 경우만 리밸런싱
                if diff_amount > 0:  # 매수 필요
                    price = self.api_client.get_current_price(ticker)
                    if price:
                        quantity = int(diff_amount / price)
                        if quantity > 0:
                            rebalance_orders.append(
                                {
                                    "ticker": ticker,
                                    "action": "buy",
                                    "quantity": quantity,
                                    "price": price,
                                }
                            )
                else:  # 매도 필요
                    quantity = self.holdings[ticker]["quantity"]
                    if quantity > 0:
                        rebalance_orders.append(
                            {
                                "ticker": ticker,
                                "action": "sell",
                                "quantity": quantity,
                                "price": self.holdings[ticker]["current_price"],
                            }
                        )

        # 리밸런싱 주문 실행
        for order in rebalance_orders:
            try:
                if order["action"] == "buy":
                    self.entry(order["ticker"], order["quantity"], order["price"])
                else:
                    self.exit(order["ticker"], reason="리밸런싱")
                time.sleep(1)  # API 호출 제한 고려
            except Exception as e:
                logger.error(f"리밸런싱 주문 실패: {order['ticker']} - {str(e)}")

        logger.info(f"포트폴리오 리밸런싱 완료: {len(rebalance_orders)}개 주문 실행")
