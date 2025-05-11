from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging
from typing import Type, Callable, Any, Optional
import time
from functools import wraps


class RetryManager:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def create_retry_decorator(
        self,
        max_attempts: int = 3,
        initial_wait: float = 1.0,
        max_wait: float = 10.0,
        exceptions: tuple = (Exception,),
        before_retry: Optional[Callable] = None,
    ) -> Callable:
        """
        재시도 데코레이터 생성

        Args:
            max_attempts: 최대 재시도 횟수
            initial_wait: 초기 대기 시간 (초)
            max_wait: 최대 대기 시간 (초)
            exceptions: 재시도할 예외 타입들
            before_retry: 재시도 전 실행할 콜백 함수
        """

        def before_sleep_callback(retry_state):
            if before_retry:
                before_retry(retry_state)
            self.logger.warning(
                f"재시도 중... (시도 {retry_state.attempt_number}/{max_attempts})"
            )

        return retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=initial_wait, max=max_wait),
            retry=retry_if_exception_type(exceptions),
            before_sleep=before_sleep_callback,
        )

    def with_session_refresh(
        self, refresh_func: Callable, max_attempts: int = 3
    ) -> Callable:
        """
        세션 리프레시가 포함된 재시도 데코레이터

        Args:
            refresh_func: 세션을 리프레시하는 함수
            max_attempts: 최대 재시도 횟수
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        if attempt == max_attempts - 1:
                            raise

                        self.logger.warning(
                            f"오류 발생: {str(e)}. 세션 리프레시 후 재시도합니다."
                        )
                        refresh_func()
                        time.sleep(1)  # 세션 리프레시 후 잠시 대기

                return None  # 이 줄은 실행되지 않아야 함

            return wrapper

        return decorator

    def create_api_retry_decorator(
        self, max_attempts: int = 3, initial_wait: float = 1.0, max_wait: float = 10.0
    ) -> Callable:
        """
        API 호출용 재시도 데코레이터

        Args:
            max_attempts: 최대 재시도 횟수
            initial_wait: 초기 대기 시간 (초)
            max_wait: 최대 대기 시간 (초)
        """

        def before_retry(retry_state):
            self.logger.warning(
                f"API 호출 실패. 재시도 중... (시도 {retry_state.attempt_number}/{max_attempts})"
            )

        return self.create_retry_decorator(
            max_attempts=max_attempts,
            initial_wait=initial_wait,
            max_wait=max_wait,
            exceptions=(ConnectionError, TimeoutError),
            before_retry=before_retry,
        )


# 사용 예시:
"""
retry_manager = RetryManager()

@retry_manager.create_api_retry_decorator()
def api_call():
    # API 호출 코드
    pass

@retry_manager.with_session_refresh(refresh_session)
def authenticated_api_call():
    # 인증이 필요한 API 호출 코드
    pass
"""
