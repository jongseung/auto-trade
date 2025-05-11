import logging
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

import config
from api.kis_api import KisAPI
from news_analyzer import NewsAnalyzer

logger = logging.getLogger("auto_trade.stock_screener")


class StockScreener:
    """주식 스크리닝 클래스"""

    def __init__(self, api_client=None):
        """
        Args:
            api_client (KisAPI): 한국투자증권 API 클라이언트
        """
        self.api_client = api_client
        self.market_codes = {"KOSPI": "J", "KOSDAQ": "K"}  # 코스피  # 코스닥
        self.candidate_stocks = []

        # 스크리닝 설정
        self.momentum_days = config.MOMENTUM_DAYS
        self.min_gap_up = config.MIN_GAP_UP
        self.min_volume_ratio = config.MIN_VOLUME_RATIO
        self.min_amount_ratio = config.MIN_AMOUNT_RATIO
        self.min_ma5_ratio = config.MIN_MA5_RATIO

        self.news_analyzer = NewsAnalyzer(
            api_client=api_client
        )  # NewsAnalyzer 인스턴스 생성

    def get_market_stocks(self, market="KOSPI"):
        """시장별 종목 리스트 조회

        Args:
            market (str): 시장 코드 ('KOSPI' 또는 'KOSDAQ')

        Returns:
            list: 종목 코드 리스트
        """
        # 실제로는 한국투자증권 API를 통해 전체 종목 리스트를 가져와야 하지만,
        # 예제에서는 임시로 몇 개의 종목만 사용
        if market == "KOSPI":
            return [
                "005930",
                "000660",
                "035420",
                "051910",
                "035720",
            ]  # 삼성전자, SK하이닉스, NAVER, LG화학, 카카오
        elif market == "KOSDAQ":
            return [
                "293490",
                "214150",
                "141080",
                "035900",
                "086520",
            ]  # 카카오게임즈, 클래시스, 레고켐바이오, JYP Ent., 에코프로
        else:
            return []

    def calculate_technical_indicators(self, ohlcv_data):
        """기술적 지표 계산

        Args:
            ohlcv_data (DataFrame): OHLCV 데이터

        Returns:
            DataFrame: 기술적 지표가 추가된 데이터프레임
        """
        df = ohlcv_data.copy()

        # 이동평균선
        df["ma5"] = df["close"].rolling(window=5).mean()
        df["ma10"] = df["close"].rolling(window=10).mean()
        df["ma20"] = df["close"].rolling(window=20).mean()

        # 이격도
        df["disparity_ma5"] = (df["close"] / df["ma5"] - 1) * 100

        # 거래량 평균
        df["volume_ma3"] = df["volume"].rolling(window=3).mean()

        # 거래대금 계산 및 평균
        df["amount_ma3"] = df["amount"].rolling(window=3).mean()

        return df

    def check_momentum(self, ohlcv_data):
        """모멘텀 조건 체크

        Args:
            ohlcv_data (DataFrame): OHLCV 데이터

        Returns:
            bool: 모멘텀 조건 충족 여부
        """
        if len(ohlcv_data) < self.momentum_days + 1:
            return False

        # 최근 N거래일 연속 종가 상승
        recent_close = ohlcv_data["close"].values[-self.momentum_days - 1 :]
        is_rising = all(
            recent_close[i] < recent_close[i + 1] for i in range(self.momentum_days)
        )

        # 당일 시초가 갭업
        if len(ohlcv_data) >= 2:
            yesterday_close = ohlcv_data["close"].values[-2]
            today_open = ohlcv_data["open"].values[-1]
            gap_up_rate = (today_open / yesterday_close) - 1
            is_gap_up = gap_up_rate >= self.min_gap_up
        else:
            is_gap_up = False

        return is_rising and is_gap_up

    def check_volume_surge(self, ohlcv_data):
        """거래량, 거래대금 급증 체크

        Args:
            ohlcv_data (DataFrame): OHLCV 데이터

        Returns:
            bool: 거래량/거래대금 급증 여부
        """
        if len(ohlcv_data) < 4:  # 최소 4일치 데이터 필요
            return False

        df = self.calculate_technical_indicators(ohlcv_data)

        try:
            # 0으로 나누기 오류 방지를 위한 안전 처리
            # 최근 3거래일 평균 대비 거래량 비율
            volume_ma3_prev = df["volume_ma3"].iloc[-2]
            if volume_ma3_prev <= 0:
                volume_ma3_prev = 1.0  # 0이면 1로 대체
            volume_ratio = df["volume"].iloc[-1] / volume_ma3_prev

            # 최근 3거래일 평균 대비 거래대금 비율
            amount_ma3_prev = df["amount_ma3"].iloc[-2]
            if amount_ma3_prev <= 0:
                amount_ma3_prev = 1.0  # 0이면 1로 대체
            amount_ratio = df["amount"].iloc[-1] / amount_ma3_prev

            is_volume_surge = volume_ratio >= self.min_volume_ratio
            is_amount_surge = amount_ratio >= self.min_amount_ratio

            return is_volume_surge and is_amount_surge
        except (IndexError, KeyError, ZeroDivisionError) as e:
            logger.warning(f"거래량 급증 체크 중 오류 발생: {str(e)}")
            return False

    def check_moving_average(self, ohlcv_data):
        """이동평균선 조건 체크

        Args:
            ohlcv_data (DataFrame): OHLCV 데이터

        Returns:
            bool: 이동평균선 조건 충족 여부
        """
        if len(ohlcv_data) < 5:  # 최소 5일치 데이터 필요 (5일선)
            return False

        df = self.calculate_technical_indicators(ohlcv_data)

        # 5일 이동평균선 위 마감
        is_above_ma5 = df["close"].iloc[-1] > df["ma5"].iloc[-1]

        # 종가/5일선 이격도 체크
        disparity_ma5 = df["disparity_ma5"].iloc[-1]
        is_disparity_ok = disparity_ma5 >= self.min_ma5_ratio * 100

        return is_above_ma5 and is_disparity_ok

    def get_selection_reason(self, ohlcv_data):
        """종목 선정 이유 파악

        Args:
            ohlcv_data (DataFrame): OHLCV 데이터

        Returns:
            str: 종목 선정 이유
        """
        reasons = []

        # 1. 모멘텀 확인
        if self.check_momentum(ohlcv_data):
            # 갭업 계산
            yesterday_close = ohlcv_data["close"].values[-2]
            today_open = ohlcv_data["open"].values[-1]
            gap_up_rate = (today_open / yesterday_close - 1) * 100

            reasons.append(f"갭업 {gap_up_rate:.1f}%")

        # 2. 거래량 급증 확인
        if len(ohlcv_data) >= 4:
            try:
                df = self.calculate_technical_indicators(ohlcv_data)

                # 거래량 급증 확인 (0으로 나누기 오류 방지)
                volume_ma3_prev = df["volume_ma3"].iloc[-2]
                if volume_ma3_prev > 0:  # 0으로 나누기 방지
                    volume_ratio = df["volume"].iloc[-1] / volume_ma3_prev
                    if volume_ratio >= self.min_volume_ratio:
                        reasons.append(f"거래량 {volume_ratio:.1f}배")
            except (IndexError, KeyError, ZeroDivisionError) as e:
                logger.warning(f"거래량 비율 계산 중 오류 발생: {str(e)}")
                pass

        # 3. 이동평균선 확인
        if len(ohlcv_data) >= 5:
            try:
                df = self.calculate_technical_indicators(ohlcv_data)

                # 이격도 확인
                disparity_ma5 = df["disparity_ma5"].iloc[-1]
                if disparity_ma5 >= self.min_ma5_ratio * 100:
                    reasons.append(f"이격도 {disparity_ma5:.1f}%")
            except (IndexError, KeyError) as e:
                logger.warning(f"이동평균선 계산 중 오류 발생: {str(e)}")
                pass

        if not reasons:
            return "기술적 지표"

        return ", ".join(reasons)

    def check_disclosure_risk(self, ticker):
        """공시 리스크 체크

        Args:
            ticker (str): 종목코드

        Returns:
            bool: 공시 리스크 없음 여부
        """
        try:
            today = datetime.now().strftime("%Y%m%d")
            disclosures = self.api_client.get_disclosure(today)

            # 리스트 형식 및 유효성 검증
            if not isinstance(disclosures, list):
                logger.warning(
                    f"공시 정보가 리스트 형식이 아닙니다: {type(disclosures)}"
                )
                return True  # 검증 실패시 리스크 없음으로 처리

            # 해당 종목의 공시 필터링 - 안전한 접근 방식 사용
            ticker_disclosures = [
                disclosure
                for disclosure in disclosures
                if disclosure.get("mksc_shrn_iscd", "") == ticker
            ]

            # 공시 취소나 지연 여부 확인
            for disclosure in ticker_disclosures:
                time = disclosure.get("dics_hora", "")
                title = disclosure.get("dics_tl", "")

                # 09:15 이전 공시 취소/지연 확인
                if time <= "0915" and ("취소" in title or "지연" in title):
                    logger.warning(f"공시 리스크 감지: {ticker} - {title}")
                    return False

            return True
        except Exception as e:
            logger.error(f"공시 리스크 확인 중 오류 발생: {str(e)}")
            return True  # 오류 발생 시 리스크 없음으로 처리

    def analyze_news_for_candidates(self, days=1):
        """뉴스 분석을 통한 종목 선정

        Args:
            days (int): 분석할 뉴스 기간(일)

        Returns:
            list: 뉴스 기반 후보 종목 리스트
        """
        logger.info(f"뉴스 기반 종목 선정 시작 (최근 {days}일)")

        try:
            # 기존 NewsAnalyzer 인스턴스 사용
            selected_stocks = self.news_analyzer.select_stocks_by_news_and_theme(
                news_days=days,
                market_cap_min=100,  # 최소 시가총액 100억
                top_n=20,  # 상위 20개 종목
                markets=["KOSPI", "KOSDAQ"],
            )

            # 결과 처리 및 변환
            news_candidates = []
            for stock in selected_stocks:
                news_candidates.append(
                    {
                        "ticker": stock["code"],
                        "name": stock["name"],
                        "score": stock.get("score", 0),
                        "market": stock.get("market", ""),
                        "reason": "뉴스 분석",
                        "is_news_pick": True,
                        "news_mentions": stock.get("news_mentions", 0),
                        "sentiment": stock.get("sentiment", 0),
                    }
                )

            logger.info(f"뉴스 분석 기반 {len(news_candidates)}개 종목 선별")
            return news_candidates

        except Exception as e:
            logger.error(f"뉴스 기반 스크리닝 중 오류 발생: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
            return []

    def calculate_stock_score(self, ticker, ohlcv_data):
        """종목 스코어 계산

        Args:
            ticker (str): 종목코드
            ohlcv_data (DataFrame): OHLCV 데이터

        Returns:
            float: 종목 스코어 (0~100)
        """
        score = 0

        # 1. 기술적 모멘텀 (30점)
        if self.check_momentum(ohlcv_data):
            score += 30

        # 2. 거래량·거래대금 급등 (40점)
        if self.check_volume_surge(ohlcv_data):
            score += 40

        # 3. 차트 지표 확인 (20점)
        if self.check_moving_average(ohlcv_data):
            score += 20

        # 4. 이벤트 리스크 점검 (10점)
        if self.check_disclosure_risk(ticker):
            score += 10

        return score

    def run_screening(self, market_list=None, include_news=True):
        """스크리닝 실행

        Args:
            market_list (list): 스크리닝할 시장 리스트 (기본값: ['KOSPI', 'KOSDAQ'])
            include_news (bool): 뉴스 기반 스크리닝 포함 여부

        Returns:
            list: 후보 종목 리스트 (스코어 내림차순 정렬)
        """
        if market_list is None:
            market_list = ["KOSPI", "KOSDAQ"]

        all_candidates = []

        # 기존 기술적 분석 기반 스크리닝
        for market in market_list:
            logger.info(f"{market} 시장 스크리닝 시작")
            stock_list = self.get_market_stocks(market)

            for ticker in stock_list:
                try:
                    # 10일치 OHLCV 데이터 조회
                    ohlcv_data = self.api_client.get_ohlcv(ticker, period="D", count=10)

                    if ohlcv_data.empty:
                        continue

                    # 종목 점수 계산
                    score = self.calculate_stock_score(ticker, ohlcv_data)

                    # 70점 이상인 경우만 후보로 선정
                    if score >= 70:
                        # 종목 기본 정보 조회
                        stock_info = self.api_client.get_stock_basic_info(ticker)
                        if not stock_info:
                            continue

                        name = stock_info.get("prdt_name", f"종목_{ticker}")
                        price = int(stock_info.get("stck_prpr", "0"))

                        all_candidates.append(
                            {
                                "ticker": ticker,
                                "name": name,
                                "score": score,
                                "price": price,
                                "reason": self.get_selection_reason(ohlcv_data),
                                "market": market,
                                "source": "technical",
                            }
                        )
                except Exception as e:
                    logger.error(f"{ticker} 스크리닝 중 오류 발생: {str(e)}")
                    continue

        # 뉴스 기반 스크리닝
        if include_news:
            try:
                news_candidates = self.analyze_news_for_candidates(days=1)
                all_candidates.extend(news_candidates)
            except Exception as e:
                logger.error(f"뉴스 기반 스크리닝 중 오류 발생: {str(e)}")

        # 스코어 기준 내림차순 정렬
        all_candidates.sort(key=lambda x: x["score"], reverse=True)

        self.candidate_stocks = all_candidates
        logger.info(f"스크리닝 완료: {len(all_candidates)}개 종목 선별됨")
        return all_candidates

    def get_final_candidates(self, limit=5):
        """최종 후보 종목 선정

        Args:
            limit (int): 최대 선정 종목 수

        Returns:
            list: 최종 후보 종목 리스트
        """
        if not self.candidate_stocks:
            return []

        # 점수 순으로 상위 N개 선택
        final_candidates = self.candidate_stocks[:limit]
        return final_candidates

    def generate_report(self, candidates=None):
        """스크리닝 결과 리포트 생성

        Args:
            candidates (list): 후보 종목 리스트 (None이면 candidate_stocks 사용)

        Returns:
            str: 리포트 내용
        """
        if candidates is None:
            candidates = self.candidate_stocks

        report = []
        report.append("=" * 80)
        report.append(
            f"주식 스크리닝 리포트 ({datetime.now().strftime('%Y-%m-%d %H:%M')})"
        )
        report.append("=" * 80)
        report.append("")

        if not candidates:
            report.append("선별된 종목이 없습니다.")
            return "\n".join(report)

        # 스크리닝 요약
        report.append("[ 스크리닝 결과 요약 ]")
        report.append("-" * 80)
        report.append(
            f"{'종목코드':<8} {'종목명':<15} {'스코어':<8} {'시장':<8} {'출처':<10} {'선정 이유'}"
        )
        report.append("-" * 80)

        for stock in candidates:
            source = "기술분석" if stock.get("source") == "technical" else "뉴스분석"
            report.append(
                f"{stock['ticker']:<8} {stock['name']:<15} {stock['score']:<8.2f} {stock['market']:<8} {source:<10} {stock['reason']}"
            )

        return "\n".join(report)
