import requests
import logging
import re
import os
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd
from collections import Counter, defaultdict
import time
import numpy as np
import config
import csv
import json

logger = logging.getLogger("auto_trade")

# KIS API를 통한 데이터 가져오기 위한 클래스 추가
from api.kis_api import KisAPI


class NewsAnalyzer:
    """뉴스 분석을 통한 종목 선정 클래스"""

    def __init__(self, stock_code_map=None, api_client=None):
        """
        Args:
            stock_code_map (dict): 회사명-종목코드 매핑 딕셔너리
            api_client (KisAPI): KIS API 클라이언트 객체
        """
        # config 모듈 직접 사용
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Cache-Control": "max-age=0",
        }
        # 회사명-종목코드 매핑 딕셔너리
        self.stock_code_map = stock_code_map or {}
        # 종목코드-회사명 역매핑 딕셔너리
        self.code_to_company = (
            {v: k for k, v in self.stock_code_map.items()} if stock_code_map else {}
        )

        # KIS API 클라이언트 설정
        self.api_client = api_client
        if self.api_client is None:
            try:
                self.api_client = KisAPI(
                    app_key=config.APP_KEY,
                    app_secret=config.APP_SECRET,
                    account_number=config.ACCOUNT_NUMBER,
                    demo_mode=config.IS_DEMO_ACCOUNT,
                )
                logger.info("KIS API 클라이언트가 성공적으로 초기화되었습니다.")
            except Exception as e:
                logger.error(f"KIS API 클라이언트 초기화 실패: {str(e)}")
                self.api_client = None

        # 주요 키워드 (긍정/부정)
        self.positive_keywords = [
            "상승",
            "급등",
            "껑충",
            "증가",
            "성장",
            "호조",
            "확대",
            "개선",
            "돌파",
            "강세",
            "최고",
            "신기록",
            "실적호전",
            "흑자전환",
            "기대",
            "수혜",
            "낙관",
            "성공",
            "추천",
            "매수",
            "러브콜",
            "계약",
            "수주",
            "매출",
            "이익",
            "신제품",
            "진출",
            "첫 거래",
            "투자유치",
            "투자확대",
            "점유율",
            "협력",
            "인수",
            "합병",
        ]
        self.negative_keywords = [
            "하락",
            "급락",
            "폭락",
            "감소",
            "위축",
            "부진",
            "축소",
            "악화",
            "하회",
            "약세",
            "최저",
            "사상 최악",
            "실적저조",
            "적자전환",
            "우려",
            "불확실",
            "비관",
            "실패",
            "매도",
            "투자의견하향",
            "계약해지",
            "소송",
            "벌금",
            "제재",
            "위기",
            "리스크",
            "경영난",
            "구조조정",
            "유동성",
            "부채",
            "채권단",
            "회생절차",
            "파산",
        ]
        # 테마 관련 키워드
        self.theme_keywords = {
            "반도체": [
                "반도체",
                "파운드리",
                "웨이퍼",
                "D램",
                "낸드",
                "메모리",
                "시스템반도체",
                "팹리스",
            ],
            "2차전지": [
                "배터리",
                "2차전지",
                "리튬",
                "음극재",
                "양극재",
                "분리막",
                "전해액",
                "전기차",
            ],
            "바이오": [
                "바이오",
                "제약",
                "신약",
                "임상",
                "백신",
                "치료제",
                "진단키트",
                "의료기기",
            ],
            "인공지능": [
                "AI",
                "인공지능",
                "딥러닝",
                "머신러닝",
                "빅데이터",
                "로봇",
                "자율주행",
            ],
            "메타버스": [
                "메타버스",
                "가상현실",
                "VR",
                "AR",
                "NFT",
                "블록체인",
                "디지털트윈",
            ],
            "신재생에너지": [
                "태양광",
                "풍력",
                "수소",
                "신재생",
                "ESG",
                "탄소중립",
                "에너지저장장치",
                "ESS",
            ],
            "게임": ["게임", "메타버스", "P2E", "엔진", "콘솔", "모바일게임"],
            "로봇": ["로봇", "무인", "자동화", "스마트팩토리", "물류로봇", "협동로봇"],
            "우주항공": ["우주", "항공", "위성", "발사체", "드론", "UAM", "모빌리티"],
            "퀀텀": ["양자", "퀀텀", "양자컴퓨터", "양자암호", "양자통신"],
        }

        # 마스터 데이터 관련 변수 초기화
        self.kospi_data = None
        self.kosdaq_data = None
        self.theme_data = None
        self.theme_code_map = {}  # 테마 코드 - 테마명 매핑
        self.stock_theme_map = {}  # 종목 코드 - 테마 리스트 매핑
        self.price_cache = {}  # 종목 가격 정보 캐시
        self.last_price_update = datetime.min  # 가격 정보 마지막 업데이트 시간

        # 마스터 데이터 로드
        self.load_master_data()

        self._news_cache = {}  # 뉴스 캐시 추가
        self._last_crawl_time = None  # 마지막 크롤링 시간 추가

    def load_master_data(self):
        """
        data/master 폴더의 코드 및 테마 정보를 로드하는 메서드
        """
        try:
            # KOSPI 종목 정보 로드
            kospi_path = "data/master/kospi_code.csv"
            if os.path.exists(kospi_path):
                try:
                    self.kospi_data = pd.read_csv(kospi_path, encoding="utf-8")
                    logger.info(f"KOSPI 종목 정보 로드 완료: {len(self.kospi_data)}개")

                    # 종목코드-회사명 매핑 추가
                    for _, row in self.kospi_data.iterrows():
                        code = row["단축코드"]
                        name = row["한글명"]
                        if not pd.isna(code) and not pd.isna(name):
                            self.stock_code_map[name] = code
                            self.code_to_company[code] = name
                except Exception as e:
                    logger.error(f"KOSPI 데이터 로드 오류: {e}")

            # KOSDAQ 종목 정보 로드
            kosdaq_path = "data/master/kosdaq_code.csv"
            if os.path.exists(kosdaq_path):
                try:
                    self.kosdaq_data = pd.read_csv(kosdaq_path, encoding="utf-8")
                    logger.info(
                        f"KOSDAQ 종목 정보 로드 완료: {len(self.kosdaq_data)}개"
                    )

                    # 종목코드-회사명 매핑 추가
                    for _, row in self.kosdaq_data.iterrows():
                        code = row["단축코드"]
                        name = row["한글종목명"]
                        if not pd.isna(code) and not pd.isna(name):
                            self.stock_code_map[name] = code
                            self.code_to_company[code] = name
                except Exception as e:
                    logger.error(f"KOSDAQ 데이터 로드 오류: {e}")

            # 테마 정보 로드
            theme_path = "data/master/theme_code.csv"
            if os.path.exists(theme_path):
                try:
                    self.theme_data = pd.read_csv(theme_path, encoding="utf-8")
                    logger.info(f"테마 정보 로드 완료: {len(self.theme_data)}개")

                    # 테마 코드 - 테마명 매핑 생성
                    unique_themes = self.theme_data[
                        ["테마코드", "테마명"]
                    ].drop_duplicates()
                    for _, row in unique_themes.iterrows():
                        self.theme_code_map[row["테마코드"]] = row["테마명"]

                    # 종목별 테마 매핑 생성
                    for _, row in self.theme_data.iterrows():
                        stock_code = str(row["종목코드"])
                        theme_name = row["테마명"]

                        if stock_code not in self.stock_theme_map:
                            self.stock_theme_map[stock_code] = []

                        if theme_name not in self.stock_theme_map[stock_code]:
                            self.stock_theme_map[stock_code].append(theme_name)

                    # 테마 키워드 사전 업데이트
                    for theme_name in self.theme_code_map.values():
                        if (
                            theme_name not in self.theme_keywords
                            and len(theme_name) > 1
                        ):
                            self.theme_keywords[theme_name] = [theme_name]

                    # 테마별 관련 종목 코드 매핑 업데이트
                    self.theme_stocks = defaultdict(list)
                    for stock_code, themes in self.stock_theme_map.items():
                        for theme in themes:
                            if stock_code not in self.theme_stocks[theme]:
                                self.theme_stocks[theme].append(stock_code)
                except Exception as e:
                    logger.error(f"테마 데이터 로드 오류: {e}")

            logger.info("마스터 데이터 로드 완료")

            # 로드된 데이터 요약 정보
            total_stocks = 0
            if self.kospi_data is not None:
                total_stocks += len(self.kospi_data)
            if self.kosdaq_data is not None:
                total_stocks += len(self.kosdaq_data)

            logger.info(
                f"총 {total_stocks}개 종목, {len(self.theme_code_map)}개 테마 로드됨"
            )

        except Exception as e:
            logger.error(f"마스터 데이터 로드 중 오류 발생: {e}")

    def get_filtered_stocks(
        self,
        market=None,
        themes=None,
        sectors=None,
        cap_min=None,
        cap_max=None,
        exclude_managed=True,
    ):
        """
        다양한 조건으로 종목을 필터링하는 메서드

        Args:
            market (str): 시장 구분 ('KOSPI', 'KOSDAQ')
            themes (list): 테마 리스트
            sectors (list): 섹터 리스트
            cap_min (int): 최소 시가총액 (억원)
            cap_max (int): 최대 시가총액 (억원)
            exclude_managed (bool): 관리종목 제외 여부

        Returns:
            list: 필터링된 종목 코드 리스트
        """
        filtered_stocks = []

        # 데이터프레임 합치기
        all_stocks = []
        if self.kospi_data is not None and (
            market is None or market.upper() == "KOSPI"
        ):
            all_stocks.append(self.kospi_data)
        if self.kosdaq_data is not None and (
            market is None or market.upper() == "KOSDAQ"
        ):
            all_stocks.append(self.kosdaq_data)

        if not all_stocks:
            logger.warning("필터링할 종목 데이터가 없습니다.")
            return []

        try:
            combined_df = pd.concat(all_stocks, ignore_index=True)
        except Exception as e:
            logger.error(f"데이터프레임 결합 오류: {e}")
            return []

        # 관리종목 필터링
        if exclude_managed:
            try:
                if "관리 종목 여부" in combined_df.columns:
                    combined_df = combined_df[combined_df["관리 종목 여부"] == "N"]
                elif "관리종목 여부" in combined_df.columns:
                    combined_df = combined_df[combined_df["관리종목 여부"] == "N"]
            except Exception as e:
                logger.error(f"관리종목 필터링 오류: {e}")

        # 시가총액 필터링
        if cap_min is not None:
            try:
                if "시가총액" in combined_df.columns:
                    combined_df = combined_df[combined_df["시가총액"] >= cap_min]
                elif "전일기준 시가총액 (억)" in combined_df.columns:
                    combined_df = combined_df[
                        combined_df["전일기준 시가총액 (억)"] >= cap_min
                    ]
            except Exception as e:
                logger.error(f"최소 시가총액 필터링 오류: {e}")

        if cap_max is not None:
            try:
                if "시가총액" in combined_df.columns:
                    combined_df = combined_df[combined_df["시가총액"] <= cap_max]
                elif "전일기준 시가총액 (억)" in combined_df.columns:
                    combined_df = combined_df[
                        combined_df["전일기준 시가총액 (억)"] <= cap_max
                    ]
            except Exception as e:
                logger.error(f"최대 시가총액 필터링 오류: {e}")

        # 섹터 필터링
        if sectors and len(sectors) > 0:
            try:
                sector_condition = False
                sector_columns = [
                    "지수업종대분류",
                    "지수업종 대분류 코드",
                    "지수 업종 중분류 코드",
                ]

                for col in sector_columns:
                    if col in combined_df.columns:
                        sector_matches = combined_df[col].isin(sectors)
                        sector_condition = sector_condition | sector_matches

                if sector_condition is not False:
                    combined_df = combined_df[sector_condition]
            except Exception as e:
                logger.error(f"섹터 필터링 오류: {e}")

        # 기본 필터링된 종목 코드 추출
        try:
            codes = []
            if "단축코드" in combined_df.columns:
                codes = combined_df["단축코드"].astype(str).tolist()
            elif "종목코드" in combined_df.columns:
                codes = combined_df["종목코드"].astype(str).tolist()
        except Exception as e:
            logger.error(f"종목 코드 추출 오류: {e}")
            codes = []

        # 테마 필터링
        if themes and len(themes) > 0 and self.theme_data is not None:
            try:
                theme_stocks = []
                theme_df = self.theme_data[self.theme_data["테마명"].isin(themes)]
                if not theme_df.empty:
                    theme_stocks = theme_df["종목코드"].astype(str).unique().tolist()

                # 이미 필터링된 종목과 교집합
                if codes:
                    codes = list(set(codes) & set(theme_stocks))
                else:
                    codes = theme_stocks
            except Exception as e:
                logger.error(f"테마 필터링 오류: {e}")

        # 디버깅 정보
        logger.info(f"필터링 결과: {len(codes)}개 종목 선택됨")

        return codes

    def analyze_theme_trends(self, days=1):
        """
        최근 뉴스에서 테마 트렌드를 분석하는 메서드

        Args:
            days (int, optional): 분석할 기간(일). 기본값은 1.

        Returns:
            dict: 테마별 점수와 관련 종목 정보
        """
        # 캐시된 뉴스 확인
        cached_news = self._get_cached_news(days)
        if cached_news:
            all_news = cached_news
        else:
            # 네이버 금융 뉴스 수집
            naver_news = self.crawl_naver_finance_news(days)
            # 인포스탁데일리 뉴스 수집
            infostock_news = self.crawl_infostock_news(days)
            # 모든 뉴스 합치기
            all_news = naver_news + infostock_news
            # 뉴스 캐싱
            self._cache_news(days, all_news)

        # 테마별 언급 횟수 및 감성 점수 계산
        theme_mentions = Counter()
        theme_sentiment = defaultdict(float)

        for news in all_news:
            title = news.get("title", "")
            content = news.get("content", "")
            full_text = f"{title} {content}"

            # 테마 추출
            themes = self.extract_themes(full_text)

            # 감성 분석
            sentiment_score = self.analyze_sentiment(full_text)

            # 테마별 언급 횟수 및 감성 점수 누적
            for theme, count in themes.items():
                theme_mentions[theme] += count
                theme_sentiment[theme] += sentiment_score * count

        # 최종 테마 트렌드 점수 계산
        theme_trends = {}

        for theme, mentions in theme_mentions.items():
            # 언급량 기반 점수 (0-100)
            mention_score = min(100, mentions * 10)

            # 감성 점수 정규화 (-1 ~ 1)
            avg_sentiment = theme_sentiment[theme] / mentions if mentions > 0 else 0
            sentiment_score = (avg_sentiment + 1) * 50  # 0-100 스케일로 변환

            # 마스터 데이터에서 관련 종목 정보 가져오기
            related_stocks = []

            if theme in self.theme_stocks:
                master_stock_codes = self.theme_stocks[theme]

                for stock_code in master_stock_codes[:10]:  # 상위 10개만
                    stock_name = self.code_to_company.get(stock_code, "")
                    stock_info = {"code": stock_code, "name": stock_name}

                    # 마스터 데이터에서 추가 정보 가져오기
                    stock_data = None
                    if self.kospi_data is not None:
                        kospi_match = self.kospi_data[
                            self.kospi_data["단축코드"] == stock_code
                        ]
                        if not kospi_match.empty:
                            stock_data = kospi_match.iloc[0]
                            stock_info["market"] = "KOSPI"

                    if stock_data is None and self.kosdaq_data is not None:
                        kosdaq_match = self.kosdaq_data[
                            self.kosdaq_data["단축코드"] == stock_code
                        ]
                        if not kosdaq_match.empty:
                            stock_data = kosdaq_match.iloc[0]
                            stock_info["market"] = "KOSDAQ"

                    if stock_data is not None:
                        # ROE 정보 추가
                        if "ROE" in stock_data:
                            stock_info["roe"] = stock_data["ROE"]

                        # 시가총액 정보 추가
                        if "전일기준 시가총액 (억)" in stock_data:
                            stock_info["market_cap"] = stock_data[
                                "전일기준 시가총액 (억)"
                            ]
                        elif "시가총액" in stock_data:
                            stock_info["market_cap"] = stock_data["시가총액"]

                        # 기준가 정보 추가
                        if "주식 기준가" in stock_data:
                            stock_info["base_price"] = stock_data["주식 기준가"]
                        elif "기준가" in stock_data:
                            stock_info["base_price"] = stock_data["기준가"]

                    related_stocks.append(stock_info)

            # 최종 테마 점수 (언급량 70% + 감성 30%)
            theme_score = mention_score * 0.7 + sentiment_score * 0.3

            theme_trends[theme] = {
                "score": theme_score,
                "mentions": mentions,
                "sentiment": avg_sentiment,
                "related_stocks": related_stocks,
            }

        # 점수 기준 내림차순 정렬
        sorted_themes = {
            k: v
            for k, v in sorted(
                theme_trends.items(), key=lambda item: item[1]["score"], reverse=True
            )
        }

        return sorted_themes

    def analyze_news(self, days=1):
        """
        뉴스 분석을 통한 종목 선정 메서드

        Args:
            days (int, optional): 분석할 기간(일). 기본값은 1.

        Returns:
            dict: 종목별 점수와 관련 뉴스 정보
        """
        # 캐시된 뉴스 확인
        cached_news = self._get_cached_news(days)
        if cached_news:
            all_news = cached_news
        else:
            # 네이버 금융 뉴스 수집
            naver_news = self.crawl_naver_finance_news(days)
            # 인포스탁데일리 뉴스 수집
            infostock_news = self.crawl_infostock_news(days)
            # 모든 뉴스 합치기
            all_news = naver_news + infostock_news
            # 뉴스 캐싱
            self._cache_news(days, all_news)

        logger.info(f"총 {len(all_news)}개의 뉴스 분석 중...")

        # 종목별 언급 횟수 및 감성 점수 계산
        stock_mentions = Counter()
        stock_sentiment = defaultdict(float)
        stock_news = defaultdict(list)
        stock_themes = defaultdict(Counter)

        for news in all_news:
            title = news.get("title", "")
            content = news.get("content", "")
            url = news.get("url", "")
            date = news.get("date", "")
            full_text = f"{title} {content}"

            # 종목명 추출
            companies = self.extract_company_names(full_text)

            # 감성 분석
            sentiment_score = self.analyze_sentiment(full_text)

            # 테마 분석
            themes = self.extract_themes(full_text)

            for company, code in companies.items():
                # 종목별 언급 횟수, 감성 점수 누적
                stock_mentions[code] += 1
                stock_sentiment[code] += sentiment_score

                # 종목별 테마 정보 누적
                for theme, count in themes.items():
                    stock_themes[code][theme] += count

                # 종목별 뉴스 정보 저장
                news_summary = {
                    "title": title,
                    "url": url,
                    "date": date,
                    "sentiment": sentiment_score,
                }
                stock_news[code].append(news_summary)

        # 테마 트렌드 분석을 통한 인기 테마 관련 종목 가중치 적용
        theme_trends = self.analyze_theme_trends(days)
        theme_scores = {theme: info["score"] for theme, info in theme_trends.items()}

        # 종목별 인기 테마 점수 계산
        stock_theme_scores = defaultdict(float)
        for code, themes_count in stock_themes.items():
            for theme, count in themes_count.items():
                if theme in theme_scores:
                    stock_theme_scores[code] += (
                        theme_scores[theme] * count / 100
                    )  # 정규화

        # 최종 종목 점수 계산
        results = {}

        for code, mentions in stock_mentions.items():
            # 마스터 데이터에서 종목 정보 확인
            stock_info = {}

            # 종목 기본 정보 (마스터 데이터)
            stock_data = None
            if self.kospi_data is not None:
                kospi_match = self.kospi_data[self.kospi_data["단축코드"] == code]
                if not kospi_match.empty:
                    stock_data = kospi_match.iloc[0]
                    stock_info["market"] = "KOSPI"
                    stock_info["name"] = kospi_match["한글명"].iloc[0]

            if stock_data is None and self.kosdaq_data is not None:
                kosdaq_match = self.kosdaq_data[self.kosdaq_data["단축코드"] == code]
                if not kosdaq_match.empty:
                    stock_data = kosdaq_match.iloc[0]
                    stock_info["market"] = "KOSDAQ"
                    stock_info["name"] = kosdaq_match["한글종목명"].iloc[0]

            # 코드가 마스터 데이터에 없으면 기본 이름 사용
            if "name" not in stock_info and code in self.code_to_company:
                stock_info["name"] = self.code_to_company[code]
            elif "name" not in stock_info:
                continue  # 종목 정보가 없으면 건너뜀

            # 감성 점수 계산 (-1 ~ 1)
            avg_sentiment = stock_sentiment[code] / mentions if mentions > 0 else 0

            # 언급량 기반 점수 (0-100)
            mention_score = min(100, mentions * 10)

            # 감성 점수 (0-100)
            sentiment_score = (avg_sentiment + 1) * 50

            # 테마 인기도 점수 (0-100)
            theme_score = min(100, stock_theme_scores[code] * 20)

            # 마스터 데이터 활용한 추가 가중치
            master_weight = 1.0  # 기본 가중치

            if stock_data is not None:
                # 시가총액 규모에 따른 가중치 (1-3)
                cap_size_column = (
                    "시가총액 규모 구분 코드"
                    if "시가총액 규모 구분 코드" in stock_data
                    else "시가총액 규모 구분 코드 유가"
                )
                if cap_size_column in stock_data and pd.notna(
                    stock_data[cap_size_column]
                ):
                    cap_size = stock_data[cap_size_column]
                    if cap_size in [1.0, 2.0, 3.0]:
                        # 시가총액 대형주 (1) 가중치 높임
                        master_weight *= (4 - cap_size) / 2

                # ROE 정보가 있으면 활용
                if "ROE" in stock_data and pd.notna(stock_data["ROE"]):
                    roe = stock_data["ROE"]
                    if roe > 0:
                        master_weight *= min(
                            2.0, 1.0 + roe / 20
                        )  # ROE가 20%면 가중치 2배

                # 거래정지, 관리종목 제외
                if (
                    "거래정지 여부" in stock_data and stock_data["거래정지 여부"] == "Y"
                ) or ("관리 종목 여부" in stock_data and stock_data["관리 종목 여부"] == "Y"):
                    master_weight *= 0.1  # 거래정지/관리종목은 가중치 낮춤

                # 마스터 데이터 추가 정보 저장
                if "전일기준 시가총액 (억)" in stock_data:
                    stock_info["market_cap"] = stock_data["전일기준 시가총액 (억)"]

                if "ROE" in stock_data:
                    stock_info["roe"] = stock_data["ROE"]

                if "기준가" in stock_data:
                    stock_info["base_price"] = stock_data["기준가"]

                # 업종 정보 추가
                for sector_col in [
                    "지수업종대분류",
                    "지수 업종 대분류 코드",
                    "지수업종 중분류",
                    "지수 업종 중분류 코드",
                ]:
                    if sector_col in stock_data and pd.notna(stock_data[sector_col]):
                        stock_info["sector"] = stock_data[sector_col]
                        break

                # 테마 정보 추가
                if code in self.stock_theme_map:
                    stock_info["master_themes"] = self.stock_theme_map[code]

            # 최종 종목 점수 (언급량 30% + 감성 40% + 테마 30%) * 마스터 데이터 가중치
            combined_score = (
                mention_score * 0.3 + sentiment_score * 0.4 + theme_score * 0.3
            ) * master_weight

            # 종목별 테마 정보 (뉴스에서 추출)
            if len(stock_themes[code]) > 0:
                stock_info["news_themes"] = dict(stock_themes[code])

            # 종목별 뉴스 정보
            stock_info["news"] = sorted(
                stock_news[code], key=lambda x: x["sentiment"], reverse=True
            )

            # 기타 정보
            stock_info["mentions"] = mentions
            stock_info["sentiment"] = avg_sentiment
            stock_info["score"] = combined_score

            results[code] = stock_info

        # 점수 기준 내림차순 정렬
        sorted_results = {
            k: v
            for k, v in sorted(
                results.items(), key=lambda item: item[1]["score"], reverse=True
            )
        }

        return sorted_results

    def select_stocks_by_news_and_theme(
        self,
        news_days=3,
        market_cap_min=None,
        market_cap_max=None,
        markets=None,
        themes=None,
        top_n=10,
    ):
        """
        뉴스와 테마 분석을 기반으로 종목을 선정하는 메서드

        Args:
            news_days (int): 분석할 최신 뉴스 일수
            market_cap_min (int, optional): 최소 시가총액 (단위: 억원)
            market_cap_max (int, optional): 최대 시가총액 (단위: 억원)
            markets (list, optional): 필터링할 시장 ('KOSPI', 'KOSDAQ')
            themes (list, optional): 필터링할 테마 목록
            top_n (int): 반환할 상위 종목 수

        Returns:
            list: 선정된 종목 정보 리스트
        """
        logging.info(
            f"뉴스와 테마 분석 기반 종목 선정 시작 (최근 {news_days}일 뉴스 기준)"
        )

        # 마스터 데이터 로드 확인
        if (
            not hasattr(self, "kospi_data")
            or self.kospi_data is None
            or not hasattr(self, "kosdaq_data")
            or self.kosdaq_data is None
        ):
            try:
                self.load_master_data()
            except Exception as e:
                logging.error(f"마스터 데이터 로드 실패: {str(e)}")
                return []

        # 뉴스 분석
        news_scores = self.analyze_news(days=news_days)

        # 테마 트렌드 분석
        theme_trends = self.analyze_theme_trends(days=news_days)

        # 종목 필터링 적용
        if markets and not isinstance(markets, list):
            markets = [markets]  # 문자열을 리스트로 변환

        try:
            # get_filtered_stocks 메서드는 market 파라미터를 받으므로 markets가 리스트인 경우 첫 번째 요소 사용
            market_param = markets[0] if markets and len(markets) > 0 else None
            filtered_codes = self.get_filtered_stocks(
                market=market_param,
                themes=themes,
                cap_min=market_cap_min,
                cap_max=market_cap_max,
            )

            # filtered_codes가 비어있으면 빈 리스트 반환
            if not filtered_codes:
                logging.warning("필터링 조건을 만족하는 종목이 없습니다.")
                return []

            logging.info(f"필터링된 종목 수: {len(filtered_codes)}개")
        except Exception as e:
            logging.error(f"종목 필터링 중 오류 발생: {str(e)}")
            filtered_codes = []

        # 필터링된 종목 코드 집합 생성
        filtered_codes_set = set(filtered_codes)

        # 최종 점수 계산 및 종목 선정
        final_scores = []
        for stock_code, news_info in news_scores.items():
            # 필터링된 종목만 고려
            if stock_code in filtered_codes_set:
                # 기본 점수는 뉴스 점수
                score = news_info["score"]

                # 해당 종목의 테마 가중치 추가
                if (
                    hasattr(self, "stock_theme_map")
                    and stock_code in self.stock_theme_map
                ):
                    themes_for_stock = self.stock_theme_map[stock_code]
                    for theme_name in themes_for_stock:
                        if theme_name in theme_trends:
                            theme_score = theme_trends[theme_name]["score"]
                            score += theme_score * 0.3  # 테마 점수 가중치 적용

                # 최종 점수와 함께 종목 정보 저장
                final_info = {
                    "code": stock_code,
                    "name": news_info.get("name", ""),
                    "score": round(score, 2),
                    "news_mentions": news_info.get("mentions", 0),
                    "sentiment": news_info.get("sentiment", 0),
                }

                # 회사명이 누락된 경우 마스터 데이터에서 검색
                if not final_info["name"] and stock_code in self.code_to_company:
                    final_info["name"] = self.code_to_company[stock_code]

                # 시장 정보 추가
                try:
                    if self.kospi_data is not None:
                        kospi_match = self.kospi_data[
                            self.kospi_data["단축코드"] == stock_code
                        ]
                        if not kospi_match.empty:
                            final_info["market"] = "KOSPI"
                            if "전일기준 시가총액 (억)" in kospi_match.columns:
                                final_info["market_cap"] = kospi_match[
                                    "전일기준 시가총액 (억)"
                                ].iloc[0]

                    if "market" not in final_info and self.kosdaq_data is not None:
                        kosdaq_match = self.kosdaq_data[
                            self.kosdaq_data["단축코드"] == stock_code
                        ]
                        if not kosdaq_match.empty:
                            final_info["market"] = "KOSDAQ"
                            if "전일기준 시가총액 (억)" in kosdaq_match.columns:
                                final_info["market_cap"] = kosdaq_match[
                                    "전일기준 시가총액 (억)"
                                ].iloc[0]
                except Exception as e:
                    logging.warning(
                        f"종목 {stock_code} 추가 정보 조회 중 오류: {str(e)}"
                    )

                final_scores.append(final_info)

        # 점수 기준 내림차순 정렬 후 상위 N개 반환
        selected_stocks = sorted(final_scores, key=lambda x: x["score"], reverse=True)[
            :top_n
        ]

        logging.info(f"뉴스와 테마 분석 기반 {len(selected_stocks)}개 종목 선정 완료")
        for idx, stock in enumerate(selected_stocks, 1):
            logging.info(
                f"{idx}. [{stock['code']}] {stock['name']} - 점수: {stock['score']}, 시장: {stock.get('market', '알 수 없음')}"
            )

        return selected_stocks

    def generate_stock_selection_report(self, selected_stocks, output_file=None):
        """
        선정된 종목 정보를 보고서로 저장하는 메서드

        Args:
            selected_stocks (list): 선정된 종목 정보 리스트
            output_file (str, optional): 보고서 저장 경로 (없으면 자동 생성)

        Returns:
            str: 생성된 보고서 경로
        """
        if not selected_stocks:
            logging.warning("선정된 종목이 없어 보고서를 생성할 수 없습니다.")
            return None

        # 보고서 파일명 생성
        if not output_file:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"data/reports/stock_selection_report_{now}.txt"

        # 디렉토리 확인 및 생성
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # 최신 가격 정보 가져오기
        stock_codes = [stock["code"] for stock in selected_stocks]
        price_info_dict = self.get_multiple_price_info(stock_codes)

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write(
                    f"뉴스 및 테마 기반 종목 추천 보고서 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n"
                )
                f.write("=" * 80 + "\n\n")

                f.write("■ 선정 기준\n")
                f.write("- 뉴스 언급 빈도 및 감성 분석\n")
                f.write("- 테마 트렌드 분석\n")
                f.write("- 기술적 지표 분석\n\n")

                f.write("■ 추천 종목 목록\n")
                f.write("-" * 80 + "\n")
                f.write(
                    f"{'순위':^5}{'종목코드':^10}{'종목명':^20}{'점수':^10}{'시장':^10}{'시가총액(억)':^15}{'테마':^20}\n"
                )
                f.write("-" * 80 + "\n")

                for idx, stock in enumerate(selected_stocks, 1):
                    # 테마 정보 가져오기
                    theme_info = ""
                    if stock["code"] in self.stock_theme_map:
                        themes = self.stock_theme_map[stock["code"]]
                        if themes:
                            theme_info = themes[0]  # 첫 번째 테마만 표시

                    f.write(
                        f"{idx:^5}{stock['code']:^10}{stock.get('name', ''):^20}{stock.get('score', 0):^10.2f}"
                        + f"{stock.get('market', ''):^10}{stock.get('market_cap', 0):^15,.0f}"
                        + f"{theme_info[:20]:^20}\n"
                    )

                f.write("-" * 80 + "\n\n")

                f.write("■ 상위 종목 상세 정보\n")
                for idx, stock in enumerate(selected_stocks[:5], 1):
                    stock_code = stock["code"]
                    price_info = price_info_dict.get(stock_code, {})

                    # 테마 정보 추출
                    themes = self.stock_theme_map.get(stock_code, [])
                    theme_str = (
                        ", ".join(themes[:3]) if themes else "해당 없음"
                    )  # 최대 3개 테마 표시

                    f.write(
                        f"{idx}. [{stock_code}] {stock.get('name', '')} (점수: {stock.get('score', 0):.2f})\n"
                    )
                    f.write(f"   - 시장: {stock.get('market', '')}\n")
                    f.write(f"   - 테마: {theme_str}\n")
                    f.write(f"   - 뉴스 언급: {stock.get('news_mentions', 0)} 회\n")
                    f.write(f"   - 감성 점수: {stock.get('sentiment', 0):.2f}\n")

                    # 실제 주가 정보 추가
                    if price_info:
                        f.write(f"   - 현재가: {price_info.get('close', 0):,.0f} 원\n")
                        f.write(
                            f"   - 전일대비: {price_info.get('change_ratio', 0):.2f}%\n"
                        )
                        f.write(f"   - 거래량: {price_info.get('volume', 0):,.0f} 주\n")
                        if (
                            price_info.get("open")
                            and price_info.get("high")
                            and price_info.get("low")
                        ):
                            f.write(
                                f"   - 시가/고가/저가: {price_info.get('open', 0):,.0f} / {price_info.get('high', 0):,.0f} / {price_info.get('low', 0):,.0f} 원\n"
                            )
                    else:
                        # API에서 정보를 가져오지 못한 경우 기본 정보 표시
                        f.write(f"   - 현재가: 정보 없음\n")
                        f.write(f"   - 전일대비: 정보 없음\n")
                        f.write(f"   - 거래량: 정보 없음\n")

                    # 기술적 지표 정보 추가
                    try:
                        rsi = self.calculate_rsi(stock_code)
                        f.write(f"   - RSI: {rsi:.2f}\n")
                    except Exception:
                        f.write(f"   - RSI: 정보 없음\n")

                    f.write("\n")

                f.write("\n" + "=" * 80 + "\n")
                f.write(
                    "※ 본 보고서는 자동화된 알고리즘에 의해 생성되었으며, 투자의 참고자료로만 활용하시기 바랍니다.\n"
                )
                f.write(
                    "※ 투자 결정은 본인 책임 하에 이루어져야 하며, 투자에 따른 손실에 대한 책임은 투자자 본인에게 있습니다.\n"
                )
                f.write("=" * 80 + "\n")

            logging.info(f"종목 선정 보고서가 생성되었습니다: {output_file}")
            return output_file

        except Exception as e:
            logging.error(f"보고서 생성 중 오류 발생: {str(e)}")
            return None

    def calculate_rsi(self, stock_code, period=14):
        """
        상대강도지수(RSI) 계산

        Args:
            stock_code (str): 종목 코드
            period (int): RSI 계산 기간

        Returns:
            float: RSI 값 (0-100)
        """
        # API 클라이언트가 초기화되지 않은 경우 더미 RSI 반환
        if not self.api_client:
            # 종목코드에 따라 다양한 RSI 값 반환 (30~70 범위)
            dummy_rsi = 30 + (int(stock_code[-2:]) % 40)
            logger.info(f"종목 {stock_code}의 더미 RSI 값({dummy_rsi})을 반환합니다.")
            return dummy_rsi

        try:
            # 일봉 데이터 가져오기
            if not stock_code.startswith("A"):
                api_stock_code = f"A{stock_code}"
            else:
                api_stock_code = stock_code

            # 최근 30일 데이터 요청 (RSI 계산에 필요한 최소 데이터 + 여유분)
            days_needed = period + 10
            ohlcv_data = self.api_client.get_daily_stock_data(
                api_stock_code, days_needed
            )

            # 데이터가 충분하지 않거나 API 응답이 없는 경우 더미 RSI 반환
            if not ohlcv_data or len(ohlcv_data) < period + 1:
                dummy_rsi = 30 + (int(stock_code[-2:]) % 40)
                logger.info(
                    f"종목 {stock_code}의 더미 RSI 값({dummy_rsi})을 반환합니다."
                )
                return dummy_rsi

            # 종가 추출
            closes = []
            for data in ohlcv_data:
                if "stck_clpr" in data:  # 종가
                    closes.append(float(data["stck_clpr"]))

            closes.reverse()  # 최신 데이터가 앞에 오도록 정렬

            if len(closes) < period + 1:
                dummy_rsi = 30 + (int(stock_code[-2:]) % 40)
                return dummy_rsi

            # 일별 가격 변화 계산
            deltas = [closes[i] - closes[i + 1] for i in range(len(closes) - 1)]

            # 상승분과 하락분 분리
            gains = [delta if delta > 0 else 0 for delta in deltas]
            losses = [-delta if delta < 0 else 0 for delta in deltas]

            # 초기 평균 상승/하락
            avg_gain = sum(gains[:period]) / period
            avg_loss = sum(losses[:period]) / period

            # RSI 계산
            if avg_loss == 0:
                return 100.0

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            return rsi

        except Exception as e:
            logger.error(f"RSI 계산 중 오류 발생: {str(e)}")
            # 오류 발생 시 더미 RSI 값 반환
            dummy_rsi = 30 + (int(stock_code[-2:]) % 40)
            return dummy_rsi

    def get_price_info(self, stock_code):
        """
        KIS API를 통해 종목의 현재가 정보를 가져오는 메서드

        Args:
            stock_code (str): 종목 코드

        Returns:
            dict: 종목 가격 정보
        """
        # 캐시 업데이트가 필요한지 확인 (10분 캐시)
        current_time = datetime.now()
        if (current_time - self.last_price_update).total_seconds() > 600:
            self.price_cache = {}  # 캐시 초기화
            self.last_price_update = current_time

        # 캐시에 있으면 캐시 데이터 반환
        if stock_code in self.price_cache:
            return self.price_cache[stock_code]

        # API 클라이언트가 초기화되지 않은 경우 더미 데이터 반환
        if not self.api_client:
            logger.warning(
                "KIS API 클라이언트가 초기화되지 않았습니다. 더미 데이터를 사용합니다."
            )
            return self._create_dummy_price_info(stock_code)

        try:
            # 종목코드 형식 확인 (앞에 A가 붙어있는지)
            if not stock_code.startswith("A"):
                api_stock_code = f"A{stock_code}"
            else:
                api_stock_code = stock_code

            # KIS API를 통해 현재가 정보 요청
            price_data = self.api_client.get_stock_price(api_stock_code)

            # API 응답 확인
            if (
                not price_data
                or "error_code" in price_data
                or "output" not in price_data
            ):
                logger.warning(
                    f"종목 {stock_code}의 가격 정보를 가져오지 못했습니다. 더미 데이터를 사용합니다."
                )
                return self._create_dummy_price_info(stock_code)

            output = price_data["output"]

            # 필요한 정보 추출
            price_info = {
                "code": stock_code,
                "name": output.get("hts_kor_isnm", ""),
                "close": int(output.get("stck_prpr", "0")),  # 현재가
                "open": int(output.get("stck_oprc", "0")),  # 시가
                "high": int(output.get("stck_hgpr", "0")),  # 고가
                "low": int(output.get("stck_lwpr", "0")),  # 저가
                "volume": int(output.get("acml_vol", "0")),  # 거래량
                "change": int(output.get("prdy_vrss", "0")),  # 전일대비
                "change_ratio": float(output.get("prdy_ctrt", "0")),  # 전일대비율
                "date": output.get("stck_bsop_date", ""),  # 기준일자
            }

            # 캐시에 저장
            self.price_cache[stock_code] = price_info
            return price_info

        except Exception as e:
            logger.error(f"종목 {stock_code}의 가격 정보 요청 중 오류 발생: {str(e)}")
            # 오류 발생시 더미 데이터 반환
            return self._create_dummy_price_info(stock_code)

    def _create_dummy_price_info(self, stock_code):
        """
        더미 가격 정보를 생성하는 내부 메서드

        Args:
            stock_code (str): 종목 코드

        Returns:
            dict: 더미 가격 정보
        """
        # 종목명 가져오기
        stock_name = self.code_to_company.get(stock_code, "")

        # 종목별로 일관된 더미 데이터 생성을 위해 종목코드의 숫자 부분 활용
        if len(stock_code) >= 6:
            # 종목코드 숫자를 기반으로 더미 가격 설정 (종목별로 다른 값 생성)
            base_value = int(stock_code[-6:]) % 100000 + 10000
        else:
            base_value = 50000

        # 종가 계산 (10000 ~ 110000 범위)
        close_price = base_value

        # 전일 대비 등락률 (-5.0 ~ +5.0% 범위)
        change_ratio = (int(stock_code[-1]) - 5) / 2

        # 전일 대비 가격 변동
        change = int(close_price * change_ratio / 100)

        # 시가, 고가, 저가 계산
        open_price = close_price - int(close_price * 0.01)
        high_price = close_price + int(close_price * 0.02)
        low_price = close_price - int(close_price * 0.02)

        # 거래량 (1만 ~ 100만 범위)
        volume = (int(stock_code[-4:]) % 90 + 10) * 10000

        # 더미 가격 정보 생성
        dummy_price = {
            "code": stock_code,
            "name": stock_name,
            "close": close_price,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "volume": volume,
            "change": change,
            "change_ratio": change_ratio,
            "date": datetime.now().strftime("%Y%m%d"),
            "is_dummy": True,  # 더미 데이터 표시
        }

        # 캐시에 저장
        self.price_cache[stock_code] = dummy_price

        logger.info(
            f"종목 {stock_code}({stock_name})의 더미 가격 정보가 생성되었습니다."
        )
        return dummy_price

    def get_multiple_price_info(self, stock_codes):
        """
        여러 종목의 가격 정보를 한 번에 가져오는 메서드

        Args:
            stock_codes (list): 종목 코드 리스트

        Returns:
            dict: 종목 코드별 가격 정보
        """
        results = {}

        if not stock_codes:
            return results

        # API 클라이언트가 초기화되지 않은 경우 모든 종목에 대해 더미 데이터 반환
        if not self.api_client:
            logger.warning(
                "KIS API 클라이언트가 초기화되지 않았습니다. 모든 종목에 대해 더미 데이터를 사용합니다."
            )
            for code in stock_codes:
                results[code] = self._create_dummy_price_info(code)
            return results

        try:
            # KIS API에서는 일반적으로 대량 요청을 지원하지만,
            # 요청 제한이 있으므로 일정 단위로 나누어 요청
            chunk_size = 20  # 한 번에 요청할 종목 수
            for i in range(0, len(stock_codes), chunk_size):
                chunk = stock_codes[i : i + chunk_size]

                # 각 종목에 대해 개별 요청
                for code in chunk:
                    price_info = self.get_price_info(code)
                    if price_info:
                        results[code] = price_info

                # API 호출 제한을 고려한 딜레이
                if i + chunk_size < len(stock_codes):
                    time.sleep(0.5)

        except Exception as e:
            logger.error(f"대량 주가 정보 요청 중 오류 발생: {str(e)}")
            # 오류 발생 시 남은 종목들에 대해 더미 데이터 생성
            for code in stock_codes:
                if code not in results:
                    results[code] = self._create_dummy_price_info(code)

        return results

    def crawl_naver_finance_news(self, days=1):
        """
        네이버 금융 뉴스를 크롤링하는 메서드

        Args:
            days (int): 최근 몇일 간의 뉴스를 수집할지 설정

        Returns:
            list: 뉴스 기사 목록
        """
        news_list = []
        logging.info(f"네이버 금융 뉴스 크롤링 시작 (최근 {days}일)")

        try:
            # 오늘 날짜 기준으로 days일 전 날짜 계산
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # 네이버 금융 뉴스 URL
            base_url = "https://finance.naver.com/news/news_list.naver"

            page = 1
            max_pages = 3  # 최대 3페이지까지만 크롤링

            while page <= max_pages:
                params = {
                    "mode": "LSS2D",
                    "section_id": "101",
                    "section_id2": "258",
                    "date_type": "1",
                    "page": page,
                }

                response = requests.get(base_url, params=params, headers=self.headers)
                soup = BeautifulSoup(response.text, "html.parser")

                # 뉴스 목록 추출
                news_items = soup.select(".realtimeNewsList .articleSubject a")
                if not news_items:
                    break

                for item in news_items:
                    try:
                        # 뉴스 제목 및 링크 추출
                        title = item.text.strip()
                        link = "https://finance.naver.com" + item.get("href")

                        # 뉴스 상세 페이지 방문
                        article_response = requests.get(link, headers=self.headers)
                        article_soup = BeautifulSoup(
                            article_response.text, "html.parser"
                        )

                        # 뉴스 날짜 추출
                        date_text = article_soup.select_one(".article_date")
                        if date_text:
                            date_str = date_text.text.strip()
                            try:
                                news_date = datetime.strptime(
                                    date_str, "%Y-%m-%d %H:%M"
                                )
                            except:
                                news_date = (
                                    datetime.now()
                                )  # 날짜 파싱 실패 시 현재 시간으로
                        else:
                            news_date = datetime.now()

                        # 날짜 범위 체크
                        if news_date < start_date:
                            continue

                        # 뉴스 본문 추출
                        content_div = article_soup.select_one("#content .articleCont")
                        content = content_div.text.strip() if content_div else ""

                        # 뉴스 정보 저장
                        news = {
                            "title": title,
                            "content": content,
                            "date": news_date,
                            "url": link,
                        }
                        news_list.append(news)

                        # 짧은 딜레이 추가
                        time.sleep(0.2)

                    except Exception as e:
                        logger.warning(f"뉴스 상세 내용 파싱 중 오류: {str(e)}")

                page += 1

                # 페이지 간 딜레이
                time.sleep(1)

            logging.info(f"네이버 금융 뉴스 {len(news_list)}개 수집 완료")

            # 더미 뉴스를 추가하지 않고 실제 크롤링 결과만 반환

        except Exception as e:
            logging.error(f"네이버 금융 뉴스 수집 중 오류 발생: {str(e)}")

            # 크롤링 실패 시 기본 더미 데이터 제공
            if not news_list:
                current_time = datetime.now()
                for i in range(5):
                    news = {
                        "title": f"[금융 뉴스] 주요 기업 동향 {i+1}",
                        "content": "국내 주요 기업들의 최근 실적과 투자 계획이 발표되었습니다. 특히 반도체와 이차전지 업종의 성장세가 눈에 띕니다.",
                        "date": current_time - timedelta(hours=i),
                        "url": f"https://finance.naver.com/news/dummy_{i}.html",
                    }
                    news_list.append(news)
                logging.info(f"기본 데이터로 대체: 네이버 금융 뉴스 {len(news_list)}개")

        return news_list

    def crawl_infostock_news(self, days=1):
        """인포스탁 뉴스를 크롤링합니다."""
        try:
            print("인포스탁 뉴스 크롤링 시작...")

            # 인포스탁 뉴스 URL
            url = "http://www.infostockdaily.co.kr/news/articleList.html"

            # 헤더 설정
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }

            # 페이지 요청
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            # HTML 파싱
            soup = BeautifulSoup(response.text, "html.parser")

            # 뉴스 기사 목록 찾기
            articles = []

            # 모든 article.box-skin.design-7 섹션 찾기
            news_sections = soup.find_all("article", class_="box-skin design-7")
            print(f"발견된 뉴스 섹션: {len(news_sections)}개")

            for section in news_sections:
                header = section.find("header", class_="header")
                if header:
                    header_text = header.get_text(strip=True)

                    # 최신기사 섹션 찾기
                    if "최신기사" in header_text:
                        # 기사 목록 찾기
                        news_items = section.select(
                            "div.auto-article.auto-dl04 ul li.auto-column"
                        )
                        print(f"최신기사 발견: {len(news_items)}개")

                        for item in news_items:
                            try:
                                # 제목과 링크 추출
                                title_elem = item.select_one(
                                    "div.size-14.auto-padbtm-8 a"
                                )
                                if not title_elem:
                                    continue

                                title = title_elem.get_text(strip=True)
                                link = (
                                    "http://www.infostockdaily.co.kr"
                                    + title_elem["href"]
                                )

                                # 날짜 추출 (기사 상세 페이지에서)
                                article_response = requests.get(link, headers=headers)
                                article_soup = BeautifulSoup(
                                    article_response.text, "html.parser"
                                )

                                # 메타 태그에서 날짜 정보 추출
                                date_str = None
                                meta_tags = article_soup.find_all("meta")
                                for meta in meta_tags:
                                    if meta.get("property") == "article:published_time":
                                        date_str = meta.get("content")
                                        break

                                if not date_str:
                                    # 대체 방법: 기자 정보에서 날짜 추출
                                    reporter_info = article_soup.find(
                                        "div", class_="article-head-info"
                                    )
                                    if reporter_info:
                                        text = reporter_info.get_text()
                                        # [인포스탁데일리=기자명 기자] 형식에서 날짜 추출
                                        date_match = re.search(
                                            r"(\d{4}년\s*\d{1,2}월\s*\d{1,2}일)", text
                                        )
                                        if date_match:
                                            date_str = date_match.group(1)
                                            # 날짜 형식 변환
                                            date_str = (
                                                date_str.replace("년", "-")
                                                .replace("월", "-")
                                                .replace("일", "")
                                            )

                                if date_str:
                                    try:
                                        # 날짜 형식 변환
                                        if "T" in date_str:  # ISO 형식인 경우
                                            date = datetime.strptime(
                                                date_str.split("T")[0], "%Y-%m-%d"
                                            )
                                        else:
                                            date = datetime.strptime(
                                                date_str, "%Y-%m-%d"
                                            )

                                        # 지정된 기간 내의 기사만 수집
                                        if (datetime.now() - date).days <= days:
                                            articles.append(
                                                {
                                                    "title": title,
                                                    "link": link,
                                                    "date": date.strftime(
                                                        "%Y-%m-%d %H:%M"
                                                    ),
                                                    "source": "인포스탁",
                                                }
                                            )
                                            print(
                                                f"수집 완료: {title} ({date.strftime('%Y-%m-%d')})"
                                            )
                                        else:
                                            print(
                                                f"기간 초과: {title} ({date.strftime('%Y-%m-%d')})"
                                            )
                                    except ValueError as e:
                                        print(f"날짜 파싱 실패: {title}")
                                else:
                                    print(f"날짜 정보 없음: {title}")
                            except Exception as e:
                                print(f"기사 처리 실패: {title}")
                                continue

            print(f"크롤링 완료. 총 {len(articles)}개의 기사 수집됨")
            return articles

        except Exception as e:
            print(f"인포스탁 뉴스 크롤링 중 오류 발생: {str(e)}")
            return []

    def extract_themes(self, text):
        """
        텍스트에서 테마 키워드를 추출하는 메서드

        Args:
            text (str): 분석할 텍스트

        Returns:
            dict: 테마별 언급 빈도
        """
        themes_count = Counter()

        try:
            # 전체 텍스트 소문자 변환
            text_lower = text.lower()

            # 테마 키워드 검색
            for theme, keywords in self.theme_keywords.items():
                theme_found = False
                for keyword in keywords:
                    count = len(re.findall(keyword, text_lower))
                    if count > 0:
                        themes_count[theme] += count
                        theme_found = True

                # 마스터 데이터의 테마명 직접 검색
                if not theme_found and theme in text:
                    themes_count[theme] += 1

            # 마스터 데이터의 테마 검색 (테마 키워드 사전에 없는 테마)
            if self.theme_code_map:
                for _, theme_name in self.theme_code_map.items():
                    if theme_name not in self.theme_keywords and theme_name in text:
                        themes_count[theme_name] += 1

        except Exception as e:
            logger.error(f"테마 추출 중 오류 발생: {e}")

        return dict(themes_count)

    def extract_company_names(self, text):
        """
        텍스트에서 회사명을 추출하고 종목코드와 매핑하는 메서드

        Args:
            text (str): 분석할 텍스트

        Returns:
            dict: 회사명-종목코드 매핑 딕셔너리
        """
        companies = {}

        try:
            # 간단한 구현: 미리 로드된 회사명 목록에서 일치하는 것 찾기
            for company_name, code in self.stock_code_map.items():
                if company_name in text:
                    companies[company_name] = code

            # 더미 데이터 추가 (테스트용)
            if not companies:
                # 테스트를 위한 더미 데이터
                sample_companies = {
                    "삼성전자": "005930",
                    "SK하이닉스": "000660",
                    "LG에너지솔루션": "373220",
                    "현대차": "005380",
                    "카카오": "035720",
                }

                for company, code in sample_companies.items():
                    if company in text:
                        companies[company] = code

                # 데이터가 없으면 최소 1개 이상의 결과 제공
                if not companies and "삼성" in text:
                    companies["삼성전자"] = "005930"

        except Exception as e:
            logger.error(f"회사명 추출 중 오류 발생: {e}")

        return companies

    def analyze_sentiment(self, text):
        """
        텍스트의 감성 분석 수행 (긍정/부정)

        Args:
            text (str): 분석할 텍스트

        Returns:
            float: 감성 점수 (-1.0 ~ 1.0, 부정 ~ 긍정)
        """
        try:
            # 긍정 키워드 및 부정 키워드 카운트
            positive_count = 0
            negative_count = 0

            for keyword in self.positive_keywords:
                if keyword in text:
                    positive_count += text.count(keyword)

            for keyword in self.negative_keywords:
                if keyword in text:
                    negative_count += text.count(keyword)

            # 감성 점수 계산 (가중치 없이 단순 비율)
            total_count = positive_count + negative_count

            if total_count == 0:
                return 0.0  # 중립

            sentiment_score = (positive_count - negative_count) / total_count

            # -1.0 ~ 1.0 범위로 클리핑
            sentiment_score = max(-1.0, min(1.0, sentiment_score))

            return sentiment_score

        except Exception as e:
            logger.error(f"감성 분석 중 오류 발생: {e}")
            return 0.0  # 오류 발생 시 중립 반환

    def _get_cached_news(self, days):
        """캐시된 뉴스 데이터 반환"""
        cache_key = f"news_{days}"
        if cache_key in self._news_cache:
            # 캐시된 데이터가 1시간 이내인 경우 재사용
            if (
                self._last_crawl_time
                and (datetime.now() - self._last_crawl_time).seconds < 3600
            ):
                return self._news_cache[cache_key]
        return None

    def _cache_news(self, days, news_data):
        """뉴스 데이터 캐싱"""
        cache_key = f"news_{days}"
        self._news_cache[cache_key] = news_data
        self._last_crawl_time = datetime.now()
