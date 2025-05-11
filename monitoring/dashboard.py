import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List
import threading
import queue
import json
import os
from dotenv import load_dotenv
import logging
from plotly.subplots import make_subplots

# 로거 설정
logger = logging.getLogger(__name__)


class Dashboard:
    """실시간 모니터링 대시보드 클래스"""

    def __init__(self, api_client, strategy):
        """
        Args:
            api_client (KisAPI): API 클라이언트
            strategy (TradingStrategy): 매매 전략 객체
        """
        self.api_client = api_client
        self.strategy = strategy
        self.data_dir = "data/dashboard"
        os.makedirs(self.data_dir, exist_ok=True)

    def update_portfolio_chart(self):
        """포트폴리오 차트 업데이트"""
        try:
            # 보유 종목 정보 가져오기
            holdings = self.strategy.get_holdings_info()

            # 차트 데이터 준비
            labels = [f"{h['name']} ({h['ticker']})" for h in holdings]
            values = [h["eval_amount"] for h in holdings]

            # 파이 차트 생성
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])

            fig.update_layout(title="포트폴리오 구성", showlegend=True)

            # 차트 저장
            fig.write_html(os.path.join(self.data_dir, "portfolio_chart.html"))

        except Exception as e:
            logger.error(f"포트폴리오 차트 업데이트 실패: {str(e)}")

    def update_performance_chart(self):
        """성과 차트 업데이트"""
        try:
            # 일별 수익률 데이터 가져오기
            daily_returns = self._get_daily_returns()

            # 차트 생성
            fig = make_subplots(
                rows=2, cols=1, subplot_titles=("일별 수익률", "누적 수익률")
            )

            # 일별 수익률
            fig.add_trace(
                go.Bar(
                    x=daily_returns.index, y=daily_returns["return"], name="일별 수익률"
                ),
                row=1,
                col=1,
            )

            # 누적 수익률
            cumulative_returns = (1 + daily_returns["return"]).cumprod() - 1
            fig.add_trace(
                go.Scatter(
                    x=daily_returns.index, y=cumulative_returns, name="누적 수익률"
                ),
                row=2,
                col=1,
            )

            fig.update_layout(height=800, title_text="투자 성과 분석")

            # 차트 저장
            fig.write_html(os.path.join(self.data_dir, "performance_chart.html"))

        except Exception as e:
            logger.error(f"성과 차트 업데이트 실패: {str(e)}")

    def _get_daily_returns(self):
        """일별 수익률 데이터 조회"""
        try:
            # 거래 이력에서 일별 수익률 계산
            history = self.strategy.order_history

            # 일별 데이터로 변환
            daily_data = {}
            for order in history:
                date = order["order_time"].date()
                if date not in daily_data:
                    daily_data[date] = {"profit_loss": 0, "trades": 0}

                if order["status"] == "filled":
                    daily_data[date]["trades"] += 1
                    if "profit_loss" in order:
                        daily_data[date]["profit_loss"] += order["profit_loss"]

            # DataFrame 생성
            df = pd.DataFrame.from_dict(daily_data, orient="index")
            df["return"] = df["profit_loss"] / 100  # 백분율을 소수로 변환

            return df

        except Exception as e:
            logger.error(f"일별 수익률 데이터 조회 실패: {str(e)}")
            return pd.DataFrame()

    def update_risk_metrics(self):
        """리스크 지표 업데이트"""
        try:
            # 일별 수익률 데이터
            daily_returns = self._get_daily_returns()

            if daily_returns.empty:
                return

            # 리스크 지표 계산
            metrics = {
                "volatility": daily_returns["return"].std() * (252**0.5),  # 연간 변동성
                "sharpe_ratio": (daily_returns["return"].mean() * 252)
                / (daily_returns["return"].std() * (252**0.5)),
                "max_drawdown": (
                    daily_returns["return"].cumsum()
                    - daily_returns["return"].cumsum().cummax()
                ).min(),
                "win_rate": (daily_returns["return"] > 0).mean(),
            }

            # 지표 저장
            with open(os.path.join(self.data_dir, "risk_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)

        except Exception as e:
            logger.error(f"리스크 지표 업데이트 실패: {str(e)}")

    def update_all(self):
        """모든 대시보드 요소 업데이트"""
        self.update_portfolio_chart()
        self.update_performance_chart()
        self.update_risk_metrics()
        logger.info("대시보드 업데이트 완료")


class TradingDashboard:
    def __init__(self):
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.data_queue = queue.Queue()
        self.account_data = self.load_account_data()
        self.setup_layout()
        self.setup_callbacks()

    def load_account_data(self) -> Dict:
        """계좌 데이터 로드"""
        try:
            # .env 파일에서 API 키 로드
            load_dotenv()

            # 계좌 데이터 파일 경로
            account_data_path = "data/account_data.json"

            if os.path.exists(account_data_path):
                with open(account_data_path, "r") as f:
                    data = json.load(f)
                    logger.info(f"계좌 데이터 로드 성공: {data}")
                    return data
            else:
                # 초기 데이터 구조 생성
                initial_data = {
                    "account": {"total_assets": 0, "cash": 0, "securities": 0},
                    "positions": {"long": 0, "short": 0},
                    "pnl": [],
                    "risk_metrics": {"var": 0, "sharpe": 0, "max_drawdown": 0},
                }
                # 파일 저장
                os.makedirs(os.path.dirname(account_data_path), exist_ok=True)
                with open(account_data_path, "w") as f:
                    json.dump(initial_data, f, indent=4)
                logger.info("초기 계좌 데이터 생성 완료")
                return initial_data
        except Exception as e:
            logger.error(f"계좌 데이터 로드 중 오류 발생: {str(e)}")
            return {
                "account": {"total_assets": 0, "cash": 0, "securities": 0},
                "positions": {"long": 0, "short": 0},
                "pnl": [],
                "risk_metrics": {"var": 0, "sharpe": 0, "max_drawdown": 0},
            }

    def setup_layout(self):
        self.app.layout = dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.H1(
                                "자동매매 시스템 모니터링 대시보드",
                                className="text-center my-4",
                            ),
                            width=12,
                        )
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H3("계좌 현황"),
                                dcc.Graph(id="account-summary"),
                                dcc.Interval(id="account-update", interval=5000),
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                html.H3("포지션 현황"),
                                dcc.Graph(id="position-summary"),
                                dcc.Interval(id="position-update", interval=5000),
                            ],
                            width=6,
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H3("P&L 현황"),
                                dcc.Graph(id="pnl-chart"),
                                dcc.Interval(id="pnl-update", interval=5000),
                            ],
                            width=12,
                        )
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H3("리스크 지표"),
                                dcc.Graph(id="risk-metrics"),
                                dcc.Interval(id="risk-update", interval=5000),
                            ],
                            width=12,
                        )
                    ]
                ),
            ],
            fluid=True,
        )

    def setup_callbacks(self):
        @self.app.callback(
            Output("account-summary", "figure"), Input("account-update", "n_intervals")
        )
        def update_account_summary(n):
            try:
                return self.create_account_summary()
            except Exception as e:
                logger.error(f"계좌 현황 업데이트 중 오류: {str(e)}")
                return self.create_empty_figure("계좌 현황")

        @self.app.callback(
            Output("position-summary", "figure"),
            Input("position-update", "n_intervals"),
        )
        def update_position_summary(n):
            try:
                return self.create_position_summary()
            except Exception as e:
                logger.error(f"포지션 현황 업데이트 중 오류: {str(e)}")
                return self.create_empty_figure("포지션 현황")

        @self.app.callback(
            Output("pnl-chart", "figure"), Input("pnl-update", "n_intervals")
        )
        def update_pnl_chart(n):
            try:
                return self.create_pnl_chart()
            except Exception as e:
                logger.error(f"P&L 차트 업데이트 중 오류: {str(e)}")
                return self.create_empty_figure("P&L 추이")

        @self.app.callback(
            Output("risk-metrics", "figure"), Input("risk-update", "n_intervals")
        )
        def update_risk_metrics(n):
            try:
                return self.create_risk_metrics()
            except Exception as e:
                logger.error(f"리스크 지표 업데이트 중 오류: {str(e)}")
                return self.create_empty_figure("리스크 지표")

    def create_empty_figure(self, title: str):
        """빈 그래프 생성"""
        return {
            "data": [],
            "layout": go.Layout(
                title=title,
                xaxis={"title": "데이터 없음"},
                yaxis={"title": "데이터 없음"},
            ),
        }

    def create_account_summary(self):
        try:
            account_data = self.account_data["account"]
            return {
                "data": [
                    go.Bar(
                        x=["총자산", "현금", "증권"],
                        y=[
                            account_data["total_assets"],
                            account_data["cash"],
                            account_data["securities"],
                        ],
                        name="계좌 현황",
                    )
                ],
                "layout": go.Layout(
                    title="계좌 현황", barmode="group", yaxis={"title": "금액 (원)"}
                ),
            }
        except Exception as e:
            logger.error(f"계좌 현황 그래프 생성 중 오류: {str(e)}")
            return self.create_empty_figure("계좌 현황")

    def create_position_summary(self):
        try:
            position_data = self.account_data["positions"]
            return {
                "data": [
                    go.Pie(
                        labels=["롱", "숏"],
                        values=[position_data["long"], position_data["short"]],
                        name="포지션 비율",
                    )
                ],
                "layout": go.Layout(title="포지션 현황"),
            }
        except Exception as e:
            logger.error(f"포지션 현황 그래프 생성 중 오류: {str(e)}")
            return self.create_empty_figure("포지션 현황")

    def create_pnl_chart(self):
        try:
            pnl_data = self.account_data["pnl"]
            if not pnl_data:
                dates = pd.date_range(start="2024-01-01", periods=1, freq="D")
                values = [0]
            else:
                dates = [pd.to_datetime(d["date"]) for d in pnl_data]
                values = [d["value"] for d in pnl_data]

            return {
                "data": [
                    go.Scatter(
                        x=dates,
                        y=values,
                        name="일별 P&L",
                    )
                ],
                "layout": go.Layout(
                    title="P&L 추이",
                    xaxis={"title": "날짜"},
                    yaxis={"title": "P&L (원)"},
                ),
            }
        except Exception as e:
            logger.error(f"P&L 차트 생성 중 오류: {str(e)}")
            return self.create_empty_figure("P&L 추이")

    def create_risk_metrics(self):
        try:
            risk_data = self.account_data["risk_metrics"]
            return {
                "data": [
                    go.Bar(
                        x=["VaR", "Sharpe Ratio", "Max Drawdown"],
                        y=[
                            risk_data["var"],
                            risk_data["sharpe"],
                            risk_data["max_drawdown"],
                        ],
                        name="리스크 지표",
                    )
                ],
                "layout": go.Layout(title="리스크 지표", yaxis={"title": "값"}),
            }
        except Exception as e:
            logger.error(f"리스크 지표 그래프 생성 중 오류: {str(e)}")
            return self.create_empty_figure("리스크 지표")

    def update_data(self, data: Dict):
        """새로운 데이터를 대시보드에 업데이트"""
        try:
            self.account_data.update(data)
            self.data_queue.put(data)

            # 데이터 파일 저장
            os.makedirs("data", exist_ok=True)
            with open("data/account_data.json", "w") as f:
                json.dump(self.account_data, f, indent=4)
            logger.info("대시보드 데이터 업데이트 완료")
        except Exception as e:
            logger.error(f"데이터 업데이트 중 오류 발생: {str(e)}")

    def run_server(self, debug=False):
        """대시보드 서버 실행"""
        logger.info("대시보드 서버 시작 (포트: 8051)")
        self.app.run_server(debug=debug, port=8051)


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # 대시보드 인스턴스 생성
    dashboard = TradingDashboard()

    # 서버 실행 (기본 포트: 8050)
    print("대시보드 서버를 시작합니다...")
    print("http://localhost:8050 에서 대시보드를 확인하실 수 있습니다.")
    dashboard.run_server(debug=True)
