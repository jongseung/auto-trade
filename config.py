import os
import yaml
from pathlib import Path
import logging
from dotenv import load_dotenv
from datetime import datetime

# .env 파일 로드
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(env_path)

# 환경변수 로드 확인을 위한 로깅
logger = logging.getLogger(__name__)
logger.info(f".env 파일 경로: {env_path}")
logger.info(f"APP_KEY: {os.getenv('APP_KEY')}")
logger.info(f"APP_SECRET: {os.getenv('APP_SECRET')}")
logger.info(f"ACCOUNT_NUMBER: {os.getenv('ACCOUNT_NUMBER')}")

# 설정 파일 경로
CONFIG_PATH = Path("config/config.yaml")


def load_config():
    """설정 파일을 로드합니다."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# 설정 로드
config = load_config()

# API 설정
API_SETTINGS = {
    "APP_KEY": os.getenv("APP_KEY"),
    "APP_SECRET": os.getenv("APP_SECRET"),
    "ACCOUNT_NUMBER": os.getenv("ACCOUNT_NUMBER"),
    "DEMO_MODE": os.getenv("DEMO_MODE", "False").lower() == "true",
    "DART_API_KEY": os.getenv(
        "DART_API_KEY", "d771554a213169e3ddc1a15422566165a6a25f9c"
    ),  # OpenDART API 키
}

# 로깅 추가
if not os.getenv("DART_API_KEY"):
    logger.warning(
        "DART_API_KEY가 설정되지 않았습니다. 공시 정보는 데모 데이터로 제공됩니다."
    )
else:
    logger.info("DART_API_KEY가 설정되었습니다.")

# 거래 설정
MAX_POSITION_SIZE = config["trading"]["max_position_size"]
MAX_DAILY_LOSS = config["trading"]["max_daily_loss"]
STOP_LOSS_PCT = config["trading"]["stop_loss_pct"]
VOLATILITY_LOOKBACK = config["trading"]["volatility_lookback"]
MAX_STOCK_RATIO = config["trading"].get(
    "max_stock_ratio", 0.1
)  # 종목당 최대 포지션 비중 (기본값 10%)
MAX_INVESTMENT_RATIO = config["trading"].get(
    "max_investment_ratio", 0.2
)  # 총 투자 비중 (기본값 20%)
LOSS_CUT_RATIO = config["trading"].get(
    "loss_cut_ratio", 0.02
)  # 기본 손절 비율 (기본값 2%)
PROFIT_CUT_RATIO = config["trading"].get(
    "profit_cut_ratio", 0.05
)  # 기본 익절 비율 (기본값 5%)
MAX_STOCK_COUNT = config["trading"].get(
    "max_stock_count", 3
)  # 최대 보유 종목 수 (기본값 3)
MAX_HOLD_DAYS = config["trading"].get("max_hold_days", 3)  # 최대 보유 일수 (기본값 3)

# 알림 설정
TELEGRAM_ENABLED = config["notifications"]["telegram"]["enabled"]
TELEGRAM_TOKEN = config["notifications"]["telegram"]["token"]
TELEGRAM_CHAT_ID = config["notifications"]["telegram"]["chat_id"]

EMAIL_ENABLED = config["notifications"]["email"]["enabled"]
EMAIL_SMTP_SERVER = config["notifications"]["email"]["smtp_server"]
EMAIL_SMTP_PORT = config["notifications"]["email"]["smtp_port"]
EMAIL_SENDER = config["notifications"]["email"]["sender"]
EMAIL_PASSWORD = config["notifications"]["email"]["password"]
EMAIL_RECIPIENTS = config["notifications"]["email"]["recipient"]

# 로깅 설정
LOG_LEVEL = config["logging"]["level"]
LOG_FILE = config["logging"]["file"]
LOG_MAX_SIZE = config["logging"]["max_size"]
LOG_BACKUP_COUNT = config["logging"]["backup_count"]

# 백테스팅 설정
BACKTEST_START_DATE = config["backtesting"]["start_date"]
BACKTEST_END_DATE = config["backtesting"]["end_date"]
BACKTEST_INITIAL_CAPITAL = config["backtesting"]["initial_capital"]
BACKTEST_COMMISSION = config["backtesting"]["commission"]
BACKTEST_SLIPPAGE = config["backtesting"]["slippage"]

# 스크리닝 설정
MIN_MARKET_CAP = config["screening"]["min_market_cap"]
MIN_VOLUME = config["screening"]["min_volume"]
MIN_VOLATILITY = config["screening"]["min_volatility"]
MIN_PRICE = config["screening"]["min_price"]
MAX_PRICE = config["screening"]["max_price"]
MOMENTUM_DAYS = config["screening"].get("momentum_days", 3)
MIN_GAP_UP = config["screening"].get("min_gap_up", 0.02)
MIN_VOLUME_RATIO = config["screening"].get("min_volume_ratio", 2.5)
MIN_AMOUNT_RATIO = config["screening"].get("min_amount_ratio", 3.0)
MIN_MA5_RATIO = config["screening"].get("min_ma5_ratio", 0.03)

# 기타 설정
TRADING_START_TIME = "09:00"
TRADING_END_TIME = "15:30"
MARKET_OPEN_TIME = "09:00"
MARKET_CLOSE_TIME = "15:30"
LUNCH_START_TIME = "11:20"
LUNCH_END_TIME = "13:00"

# 데이터 디렉토리
DATA_DIR = Path("data")
LOG_DIR = Path("logs")
BACKTEST_DIR = Path("backtest_results")

# 디렉토리 생성
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
BACKTEST_DIR.mkdir(exist_ok=True)


# 로깅 설정 함수
def setup_logging():
    """로깅 설정"""
    # 로그 디렉토리 생성
    os.makedirs(LOG_DIR, exist_ok=True)

    # 로그 파일명 설정
    today = datetime.now().strftime("%Y%m%d")
    log_file = f"{LOG_DIR}/auto_trade_{today}.log"

    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, LOG_LEVEL))

    # 기존 핸들러 제거
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 파일 핸들러
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(file_handler)

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(console_handler)

    return root_logger


# 로깅 설정
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# 기타 설정
BACKTEST_MODE = os.getenv("BACKTEST_MODE", "False").lower() == "true"

# 전략 관련 추가 상수
TAKE_PROFIT_RATIO = PROFIT_CUT_RATIO
STOP_LOSS_RATIO = LOSS_CUT_RATIO
TRAILING_STOP = 0.03  # 트레일링 스탑 기본값 3%
MAX_STOCKS = MAX_STOCK_COUNT
PORTFOLIO_RATIO = MAX_INVESTMENT_RATIO
STOCK_RATIO = MAX_STOCK_RATIO

MORNING_ENTRY_START = "09:00"
MORNING_ENTRY_END = "09:05"
ADDITIONAL_ENTRY_END = "09:10"
MARKET_CLOSE = "15:30"

# 캐시 설정
CACHE_CONFIG = {
    "CACHE_TYPE": "filesystem",
    "CACHE_DIR": "cache",
    "CACHE_DEFAULT_TIMEOUT": 1800,  # 30분
    "CACHE_THRESHOLD": 1000,
}
