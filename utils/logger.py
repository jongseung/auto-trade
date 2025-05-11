import logging
import logging.handlers
import os
import yaml
from datetime import datetime
from typing import Optional
import json


def setup_logging(config_path: str = "config/config.yaml") -> logging.Logger:
    """로깅 시스템 설정"""
    try:
        # 설정 파일 로드
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        log_config = config.get("logging", {})

        # 로그 디렉토리 생성
        log_dir = os.path.dirname(log_config.get("file", "logs/auto_trade.log"))
        os.makedirs(log_dir, exist_ok=True)

        # 로거 설정
        logger = logging.getLogger("auto_trade")
        logger.setLevel(getattr(logging, log_config.get("level", "INFO")))

        # 파일 핸들러 설정 (로테이션)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_config.get("file", "logs/auto_trade.log"),
            maxBytes=log_config.get("max_size", 10 * 1024 * 1024),  # 10MB
            backupCount=log_config.get("backup_count", 5),
            encoding="utf-8",
        )

        # 콘솔 핸들러 설정
        console_handler = logging.StreamHandler()

        # 포맷터 설정
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 핸들러 추가
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # 로깅 시작 메시지
        logger.info("로깅 시스템이 초기화되었습니다.")

        return logger

    except Exception as e:
        print(f"로깅 시스템 초기화 실패: {str(e)}")
        # 기본 로거 반환
        return logging.getLogger("auto_trade")


def log_trade(logger: logging.Logger, trade_info: dict):
    """거래 로깅"""
    logger.info(f"거래 실행: {json.dumps(trade_info, ensure_ascii=False)}")


def log_error(logger: logging.Logger, error_info: dict):
    """에러 로깅"""
    logger.error(f"에러 발생: {json.dumps(error_info, ensure_ascii=False)}")


def log_portfolio(logger: logging.Logger, portfolio_info: dict):
    """포트폴리오 상태 로깅"""
    logger.info(f"포트폴리오 상태: {json.dumps(portfolio_info, ensure_ascii=False)}")


def log_market_data(logger: logging.Logger, market_data: dict):
    """시장 데이터 로깅"""
    logger.debug(f"시장 데이터: {json.dumps(market_data, ensure_ascii=False)}")


def log_performance(logger: logging.Logger, performance_metrics: dict):
    """성과 지표 로깅"""
    logger.info(f"성과 지표: {json.dumps(performance_metrics, ensure_ascii=False)}")
