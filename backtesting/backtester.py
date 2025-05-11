import logging
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
from pathlib import Path
import plotly.graph_objects as go

import config
from api.kis_api import KisAPI
from screener.stock_screener import StockScreener
from news_analyzer import NewsAnalyzer

logger = logging.getLogger("auto_trade.backtester")


class BackTester:
    """단기매매 전략 백테스팅 클래스"""

    def __init__(self, api_client, start_date=None, end_date=None):
        """
        Args:
            api_client (KisAPI): 한국투자증권 API 클라이언트
            start_date (str): 백테스팅 시작일 (YYYYMMDD 형식)
            end_date (str): 백테스팅 종료일 (YYYYMMDD 형식)
        """
        self.api_client = api_client
        self.screener = StockScreener(api_client)
        self.news_analyzer = NewsAnalyzer()

        # 백테스팅 기간 설정
        self.start_date = start_date or (datetime.now() - timedelta(days=30)).strftime(
            "%Y%m%d"
        )
        self.end_date = end_date or datetime.now().strftime("%Y%m%d")

        # 백테스팅 결과 저장 디렉토리
        self.result_dir = "data/backtest"
        Path(self.result_dir).mkdir(parents=True, exist_ok=True)

        # 백테스팅 설정
        self.max_stocks = config.MAX_STOCK_COUNT
        self.profit_cut_ratio = config.PROFIT_CUT_RATIO
        self.loss_cut_ratio = config.LOSS_CUT_RATIO
        self.max_hold_days = config.MAX_HOLD_DAYS
        self.initial_capital = 10000000  # 초기 자본금 (1,000만원)

        # 백테스팅 결과 저장용 변수들
        self.portfolio = {
            "cash": self.initial_capital,
            "stocks": {},
            "history": [],
            "daily_nav": [],
        }

        # 휴장일 데이터
        self.holidays = self._load_holidays()

    def _load_holidays(self):
        """휴장일 데이터 로드"""
        try:
            holiday_path = "data/master/market_holidays.csv"
            if os.path.exists(holiday_path):
                holidays_df = pd.read_csv(holiday_path)
                return holidays_df["일자"].astype(str).tolist()
            else:
                logger.warning("휴장일 데이터 파일이 없습니다. 주말만 제외합니다.")
                return []
        except Exception as e:
            logger.error(f"휴장일 데이터 로드 오류: {e}")
            return []

    def _is_trading_day(self, date_str):
        """거래일 여부 확인"""
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        # 주말 체크
        if date_obj.weekday() >= 5:  # 5: 토요일, 6: 일요일
            return False
        # 휴장일 체크
        if date_str in self.holidays:
            return False
        return True

    def _get_trading_days(self):
        """백테스팅 기간 중 거래일 리스트 반환"""
        start = datetime.strptime(self.start_date, "%Y%m%d")
        end = datetime.strptime(self.end_date, "%Y%m%d")
        delta = end - start

        trading_days = []
        for i in range(delta.days + 1):
            date = start + timedelta(days=i)
            date_str = date.strftime("%Y%m%d")
            if self._is_trading_day(date_str):
                trading_days.append(date_str)

        return trading_days

    def _calculate_nav(self, date):
        """포트폴리오 순자산가치(NAV) 계산"""
        nav = self.portfolio["cash"]

        for ticker, position in self.portfolio["stocks"].items():
            # 해당 종목의 해당 일자 종가 가져오기
            try:
                ohlcv = self.api_client.get_ohlcv(ticker, "D", 1)
                if ohlcv is not None and not ohlcv.empty:
                    # date 컬럼이 있으면 해당 날짜만 필터링
                    if "date" in ohlcv.columns:
                        ohlcv = ohlcv[ohlcv["date"] == pd.to_datetime(date)]
                    if not ohlcv.empty:
                        current_price = ohlcv["close"].iloc[-1]
                        nav += position["quantity"] * current_price
            except Exception as e:
                logger.error(f"NAV 계산 중 오류 발생: {ticker}, {date}, {e}")

        # 일별 NAV 기록
        self.portfolio["daily_nav"].append(
            {
                "date": date,
                "nav": nav,
                "return_pct": (nav / self.initial_capital - 1) * 100,
            }
        )

        return nav

    def _execute_buy(self, ticker, price, date, reason):
        """매수 실행 (백테스팅용)"""
        cash = self.portfolio["cash"]

        # 최대 보유 종목 수 확인
        if len(self.portfolio["stocks"]) >= self.max_stocks:
            logger.info(f"최대 보유 종목 수에 도달했습니다. 매수 실행 불가: {ticker}")
            return False

        # 한 종목당 최대 투자금 계산 (초기 자본의 5%)
        max_per_stock = self.initial_capital * config.MAX_STOCK_RATIO

        # 매수할 수량 계산
        available_cash = min(cash, max_per_stock)
        quantity = int(available_cash / price)

        if quantity == 0:
            logger.info(f"매수 가능 수량이 0입니다. 매수 실행 불가: {ticker}")
            return False

        total_amount = quantity * price

        # 포트폴리오 업데이트
        self.portfolio["cash"] -= total_amount

        # 해당 종목이 이미 있는 경우 평균단가 계산
        if ticker in self.portfolio["stocks"]:
            current_position = self.portfolio["stocks"][ticker]
            current_quantity = current_position["quantity"]
            current_price = current_position["price"]

            new_quantity = current_quantity + quantity
            new_price = (
                current_quantity * current_price + quantity * price
            ) / new_quantity

            self.portfolio["stocks"][ticker] = {
                "quantity": new_quantity,
                "price": new_price,
                "buy_date": current_position["buy_date"],
            }
        else:
            self.portfolio["stocks"][ticker] = {
                "quantity": quantity,
                "price": price,
                "buy_date": date,
            }

        # 매매 내역 기록
        self.portfolio["history"].append(
            {
                "date": date,
                "ticker": ticker,
                "action": "BUY",
                "quantity": quantity,
                "price": price,
                "total": total_amount,
                "reason": reason,
            }
        )

        logger.info(
            f"매수 실행: {ticker}, {quantity}주, 단가: {price}, 총액: {total_amount}, 사유: {reason}"
        )
        return True

    def _execute_sell(self, ticker, price, date, reason):
        """매도 실행 (백테스팅용)"""
        if ticker not in self.portfolio["stocks"]:
            logger.warning(f"보유하지 않은 종목 매도 시도: {ticker}")
            return False

        position = self.portfolio["stocks"][ticker]
        quantity = position["quantity"]
        buy_price = position["price"]

        total_amount = quantity * price
        profit_loss = total_amount - (quantity * buy_price)
        profit_loss_pct = (price / buy_price - 1) * 100

        # 포트폴리오 업데이트
        self.portfolio["cash"] += total_amount
        del self.portfolio["stocks"][ticker]

        # 매매 내역 기록
        self.portfolio["history"].append(
            {
                "date": date,
                "ticker": ticker,
                "action": "SELL",
                "quantity": quantity,
                "price": price,
                "total": total_amount,
                "profit_loss": profit_loss,
                "profit_loss_pct": profit_loss_pct,
                "reason": reason,
            }
        )

        logger.info(
            f"매도 실행: {ticker}, {quantity}주, 단가: {price}, 총액: {total_amount}, 손익: {profit_loss}, 손익률: {profit_loss_pct:.2f}%, 사유: {reason}"
        )
        return True

    def _check_sell_conditions(self, ticker, current_price, current_date):
        """매도 조건 체크"""
        if ticker not in self.portfolio["stocks"]:
            return False, None

        position = self.portfolio["stocks"][ticker]
        buy_price = position["buy_price"]
        buy_date = position["buy_date"]

        # 수익률 계산
        profit_pct = (current_price / buy_price - 1) * 100

        # 1. 익절 조건
        if profit_pct >= self.profit_cut_ratio * 100:
            return True, f"익절 {profit_pct:.2f}%"

        # 2. 손절 조건
        if profit_pct <= -self.loss_cut_ratio * 100:
            return True, f"손절 {profit_pct:.2f}%"

        # 3. 최대 보유 기간 체크
        buy_date_obj = datetime.strptime(buy_date, "%Y%m%d")
        current_date_obj = datetime.strptime(current_date, "%Y%m%d")
        hold_days = (current_date_obj - buy_date_obj).days

        if hold_days >= self.max_hold_days:
            return True, f"보유 기간 초과 ({hold_days}일)"

        return False, None

    def run_backtest(self):
        """백테스팅 실행"""
        logger.info(f"백테스팅 시작: {self.start_date} ~ {self.end_date}")

        # 초기화
        self.portfolio = {
            "cash": self.initial_capital,
            "stocks": {},
            "history": [],
            "daily_nav": [],
        }

        # 거래일 리스트 가져오기
        trading_days = self._get_trading_days()
        logger.info(f"백테스팅 거래일 수: {len(trading_days)}일")

        for day_idx, current_date in enumerate(trading_days):
            logger.info(
                f"백테스팅 진행 중: {current_date} ({day_idx+1}/{len(trading_days)}일)"
            )

            # 1. 기존 포지션 체크 (매도 조건 확인)
            tickers_to_sell = []
            for ticker in list(self.portfolio["stocks"].keys()):
                try:
                    # 해당 종목의 당일 OHLCV 데이터 가져오기
                    ohlcv = self.api_client.get_ohlcv(ticker, "D", 1)
                    if ohlcv is None or ohlcv.empty:
                        continue
                    if "date" in ohlcv.columns:
                        ohlcv = ohlcv[ohlcv["date"] == pd.to_datetime(current_date)]
                    if ohlcv.empty:
                        continue
                    current_price = ohlcv["close"].iloc[-1]
                    should_sell, reason = self._check_sell_conditions(
                        ticker, current_price, current_date
                    )

                    if should_sell:
                        tickers_to_sell.append((ticker, current_price, reason))
                except Exception as e:
                    logger.error(
                        f"매도 조건 체크 중 오류 발생: {ticker}, {current_date}, {e}"
                    )

            # 매도 실행
            for ticker, price, reason in tickers_to_sell:
                self._execute_sell(ticker, price, current_date, reason)

            # 2. 신규 매수 후보 스크리닝
            try:
                # 스크리닝 실행 (실제 로직은 간소화)
                candidates = self.screener.run_screening(
                    market_list=["KOSPI", "KOSDAQ"], include_news=True
                )

                # 최종 후보 가져오기
                top_candidates = self.screener.get_final_candidates(limit=10)

                # 현재 보유 종목 수가 최대치 미만인 경우에만 매수 진행
                if len(self.portfolio["stocks"]) < self.max_stocks and top_candidates:
                    for candidate in top_candidates:
                        ticker = candidate["ticker"]

                        # 이미 보유 중인 종목은 건너뛰기
                        if ticker in self.portfolio["stocks"]:
                            continue

                        # 해당 종목의 당일 OHLCV 데이터 가져오기
                        ohlcv = self.api_client.get_ohlcv(ticker, "D", 1)
                        if ohlcv is None or ohlcv.empty:
                            continue
                        if "date" in ohlcv.columns:
                            ohlcv = ohlcv[ohlcv["date"] == pd.to_datetime(current_date)]
                        if ohlcv.empty:
                            continue
                        current_price = ohlcv["close"].iloc[-1]
                        reason = f"스크리닝 점수: {candidate['score']:.2f}"

                        # 매수 실행
                        result = self._execute_buy(
                            ticker, current_price, current_date, reason
                        )

                        # 최대 보유 종목 수에 도달하면 중단
                        if result and len(self.portfolio["stocks"]) >= self.max_stocks:
                            break
            except Exception as e:
                logger.error(f"종목 스크리닝 중 오류 발생: {current_date}, {e}")

            # 3. 일별 NAV 계산
            self._calculate_nav(current_date)

        # 백테스팅 결과 저장
        self._save_backtest_results()

        # 결과 분석
        return self._analyze_backtest_results()

    def _save_backtest_results(self):
        """백테스팅 결과 저장"""
        # 결과 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"{self.result_dir}/backtest_result_{timestamp}.json"

        # 결과 데이터 구성
        result_data = {
            "backtest_period": {
                "start_date": self.start_date,
                "end_date": self.end_date,
            },
            "parameters": {
                "initial_capital": self.initial_capital,
                "max_stocks": self.max_stocks,
                "profit_cut_ratio": self.profit_cut_ratio,
                "loss_cut_ratio": self.loss_cut_ratio,
                "max_hold_days": self.max_hold_days,
            },
            "trade_history": self.portfolio["history"],
            "daily_nav": self.portfolio["daily_nav"],
        }

        # JSON 파일로 저장
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        logger.info(f"백테스팅 결과 저장 완료: {result_file}")
        return result_file

    def _analyze_backtest_results(self):
        """백테스팅 결과 분석"""
        if not self.portfolio["daily_nav"]:
            logger.error("백테스팅 결과가 없습니다.")
            return None

        # 결과 데이터프레임 생성
        nav_df = pd.DataFrame(self.portfolio["daily_nav"])

        # 거래 내역 데이터프레임 생성
        if self.portfolio["history"]:
            trade_df = pd.DataFrame(self.portfolio["history"])
        else:
            trade_df = pd.DataFrame(
                columns=[
                    "date",
                    "ticker",
                    "action",
                    "quantity",
                    "price",
                    "total",
                    "profit_loss",
                    "profit_loss_pct",
                ]
            )

        # 최종 성과 계산
        initial_capital = self.initial_capital
        final_nav = nav_df["nav"].iloc[-1] if not nav_df.empty else initial_capital

        total_return = final_nav - initial_capital
        total_return_pct = (final_nav / initial_capital - 1) * 100

        # 연 수익률 계산
        start_date = datetime.strptime(self.start_date, "%Y%m%d")
        end_date = datetime.strptime(self.end_date, "%Y%m%d")
        years = (end_date - start_date).days / 365

        if years > 0:
            annual_return = ((1 + total_return_pct / 100) ** (1 / years) - 1) * 100
        else:
            annual_return = total_return_pct

        # 승률 계산
        winning_trades = trade_df[trade_df["action"] == "SELL"][
            trade_df["profit_loss"] > 0
        ]
        total_trades = trade_df[trade_df["action"] == "SELL"]

        win_rate = (
            len(winning_trades) / len(total_trades) * 100
            if len(total_trades) > 0
            else 0
        )

        # 최대 낙폭(MDD) 계산
        rolling_max = nav_df["nav"].cummax()
        drawdown = (nav_df["nav"] / rolling_max - 1) * 100
        mdd = drawdown.min()

        # 샤프 비율 계산 (간소화된 방식)
        daily_returns = nav_df["nav"].pct_change().dropna()
        sharpe_ratio = (
            np.sqrt(252) * daily_returns.mean() / daily_returns.std()
            if len(daily_returns) > 0
            else 0
        )

        # 결과 정리
        results = {
            "initial_capital": initial_capital,
            "final_nav": final_nav,
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "annual_return": annual_return,
            "win_rate": win_rate,
            "mdd": mdd,
            "sharpe_ratio": sharpe_ratio,
            "total_trades": len(total_trades),
            "avg_profit_per_trade": (
                winning_trades["profit_loss"].mean() if len(winning_trades) > 0 else 0
            ),
            "avg_loss_per_trade": (
                trade_df[trade_df["action"] == "SELL"][trade_df["profit_loss"] < 0][
                    "profit_loss"
                ].mean()
                if len(
                    trade_df[trade_df["action"] == "SELL"][trade_df["profit_loss"] < 0]
                )
                > 0
                else 0
            ),
        }

        # 결과 출력
        logger.info("===== 백테스팅 결과 =====")
        logger.info(f"기간: {self.start_date} ~ {self.end_date}")
        logger.info(f"초기 자본금: {initial_capital:,.0f}원")
        logger.info(f"최종 자산가치: {final_nav:,.0f}원")
        logger.info(f"총 수익률: {total_return_pct:.2f}%")
        logger.info(f"연 수익률: {annual_return:.2f}%")
        logger.info(f"총 거래 횟수: {len(total_trades)}회")
        logger.info(f"승률: {win_rate:.2f}%")
        logger.info(f"최대 낙폭(MDD): {mdd:.2f}%")
        logger.info(f"샤프 비율: {sharpe_ratio:.2f}")

        return results

    def plot_performance(self, output_file=None):
        """백테스팅 결과 시각화"""
        if not self.portfolio["daily_nav"]:
            logger.error("백테스팅 결과가 없습니다.")
            return None

        nav_df = pd.DataFrame(self.portfolio["daily_nav"])
        nav_df["date"] = pd.to_datetime(nav_df["date"])
        nav_df.set_index("date", inplace=True)

        # 그림 크기 설정
        plt.figure(figsize=(12, 8))

        # 1. 자산가치 추이
        plt.subplot(2, 1, 1)
        plt.plot(nav_df.index, nav_df["nav"], "b-", label="Portfolio Value")
        plt.title("Portfolio Value Over Time")
        plt.grid(True)
        plt.legend()

        # 2. 수익률 추이
        plt.subplot(2, 1, 2)
        plt.plot(nav_df.index, nav_df["return_pct"], "g-", label="Return (%)")
        plt.axhline(y=0, color="r", linestyle="-", alpha=0.3)
        plt.title("Portfolio Return (%)")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()

        # 파일로 저장
        if output_file:
            plt.savefig(output_file)
            logger.info(f"성과 차트 저장 완료: {output_file}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{self.result_dir}/backtest_chart_{timestamp}.png"
            plt.savefig(output_file)
            logger.info(f"성과 차트 저장 완료: {output_file}")

        plt.close()
        return output_file

    def visualize_results(self):
        """백테스팅 결과 시각화"""
        try:
            # 결과 디렉토리 생성
            result_dir = os.path.join(
                self.result_dir, f"backtest_{self.start_date}_{self.end_date}"
            )
            os.makedirs(result_dir, exist_ok=True)

            # 1. 포트폴리오 가치 변화 차트
            self._plot_portfolio_value(result_dir)

            # 2. 수익률 분포 차트
            self._plot_returns_distribution(result_dir)

            # 3. 월별 수익률 히트맵
            self._plot_monthly_returns_heatmap(result_dir)

            # 4. 종목별 수익률 차트
            self._plot_stock_returns(result_dir)

            logger.info(f"백테스팅 결과 시각화 완료: {result_dir}")

        except Exception as e:
            logger.error(f"백테스팅 결과 시각화 실패: {str(e)}")

    def _plot_portfolio_value(self, result_dir):
        """포트폴리오 가치 변화 차트 생성"""
        try:
            # 일별 NAV 데이터
            dates = [entry["date"] for entry in self.portfolio["daily_nav"]]
            values = [entry["nav"] for entry in self.portfolio["daily_nav"]]

            # 차트 생성
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=dates, y=values, mode="lines", name="포트폴리오 가치")
            )

            # 레이아웃 설정
            fig.update_layout(
                title="포트폴리오 가치 변화",
                xaxis_title="날짜",
                yaxis_title="포트폴리오 가치 (원)",
                showlegend=True,
            )

            # 차트 저장
            fig.write_html(os.path.join(result_dir, "portfolio_value.html"))

        except Exception as e:
            logger.error(f"포트폴리오 가치 차트 생성 실패: {str(e)}")

    def _plot_returns_distribution(self, result_dir):
        """수익률 분포 차트 생성"""
        try:
            # 일별 수익률 계산
            daily_returns = []
            for i in range(1, len(self.portfolio["daily_nav"])):
                prev_nav = self.portfolio["daily_nav"][i - 1]["nav"]
                curr_nav = self.portfolio["daily_nav"][i]["nav"]
                daily_return = (curr_nav / prev_nav) - 1
                daily_returns.append(daily_return)

            # 히스토그램 생성
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=daily_returns, nbinsx=50, name="일별 수익률"))

            # 레이아웃 설정
            fig.update_layout(
                title="일별 수익률 분포",
                xaxis_title="수익률",
                yaxis_title="빈도",
                showlegend=True,
            )

            # 차트 저장
            fig.write_html(os.path.join(result_dir, "returns_distribution.html"))

        except Exception as e:
            logger.error(f"수익률 분포 차트 생성 실패: {str(e)}")

    def _plot_monthly_returns_heatmap(self, result_dir):
        """월별 수익률 히트맵 생성"""
        try:
            # 월별 수익률 데이터 준비
            monthly_returns = {}
            for entry in self.portfolio["daily_nav"]:
                date = entry["date"]
                year = date.year
                month = date.month

                if (year, month) not in monthly_returns:
                    monthly_returns[(year, month)] = []
                monthly_returns[(year, month)].append(entry["nav"])

            # 월별 수익률 계산
            heatmap_data = []
            for (year, month), navs in monthly_returns.items():
                if len(navs) > 1:
                    monthly_return = (navs[-1] / navs[0]) - 1
                    heatmap_data.append(
                        {"year": year, "month": month, "return": monthly_return}
                    )

            # 히트맵 데이터 변환
            df = pd.DataFrame(heatmap_data)
            pivot_table = df.pivot(index="year", columns="month", values="return")

            # 히트맵 생성
            fig = go.Figure(
                data=go.Heatmap(
                    z=pivot_table.values,
                    x=pivot_table.columns,
                    y=pivot_table.index,
                    colorscale="RdYlGn",
                )
            )

            # 레이아웃 설정
            fig.update_layout(
                title="월별 수익률 히트맵", xaxis_title="월", yaxis_title="년도"
            )

            # 차트 저장
            fig.write_html(os.path.join(result_dir, "monthly_returns_heatmap.html"))

        except Exception as e:
            logger.error(f"월별 수익률 히트맵 생성 실패: {str(e)}")

    def _plot_stock_returns(self, result_dir):
        """종목별 수익률 차트 생성"""
        try:
            # 종목별 수익률 데이터 준비
            stock_returns = {}
            for trade in self.portfolio["history"]:
                if trade["type"] == "sell":
                    ticker = trade["ticker"]
                    if ticker not in stock_returns:
                        stock_returns[ticker] = []
                    stock_returns[ticker].append(trade["profit_loss"])

            # 종목별 평균 수익률 계산
            avg_returns = {
                ticker: sum(returns) / len(returns)
                for ticker, returns in stock_returns.items()
            }

            # 차트 생성
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=list(avg_returns.keys()),
                    y=list(avg_returns.values()),
                    name="평균 수익률",
                )
            )

            # 레이아웃 설정
            fig.update_layout(
                title="종목별 평균 수익률",
                xaxis_title="종목",
                yaxis_title="평균 수익률 (%)",
                showlegend=True,
            )

            # 차트 저장
            fig.write_html(os.path.join(result_dir, "stock_returns.html"))

        except Exception as e:
            logger.error(f"종목별 수익률 차트 생성 실패: {str(e)}")
