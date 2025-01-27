from typing import Optional, Type, Any
import logging
import functools
import time
from requests.exceptions import RequestException, HTTPError

__all__ = ["retry_with_backoff", "handle_api_errors", "RateLimiter"]

logger = logging.getLogger(__name__)

class KaggleAPIError(Exception):
    """Base exception for Kaggle API errors"""
    def __init__(self, message: str, status_code: Optional[int] = None, response: Any = None):
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(self.message)

class AuthenticationError(KaggleAPIError):
    """Authentication failed"""
    pass

class RateLimitError(KaggleAPIError):
    """Rate limit exceeded"""
    pass

class ResourceNotFoundError(KaggleAPIError):
    """Requested resource not found"""
    pass

class ValidationError(KaggleAPIError):
    """Invalid request parameters"""
    pass

def handle_api_errors(func):
    """Decorator to handle API errors"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except HTTPError as e:
            status_code = e.response.status_code
            if status_code == 401:
                raise AuthenticationError("Authentication failed", status_code, e.response)
            elif status_code == 403:
                raise RateLimitError("Rate limit exceeded", status_code, e.response)
            elif status_code == 404:
                raise ResourceNotFoundError("Resource not found", status_code, e.response)
            elif status_code == 422:
                raise ValidationError("Invalid request", status_code, e.response)
            raise KaggleAPIError(f"API error: {str(e)}", status_code, e.response)
        except RequestException as e:
            raise KaggleAPIError(f"Request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            raise

def retry_with_backoff(
    max_retries: int = 3,
    backoff_factor: float = 1.5,
    error_types: tuple = (RequestException,)
):
    """Retry decorator with exponential backoff"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except error_types as e:
                    last_error = e
                    if attempt == max_retries - 1:
                        raise
                    wait_time = backoff_factor * (2 ** attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {str(e)}. "
                        f"Retrying in {wait_time} seconds..."
                    )
                    time.sleep(wait_time)
            raise last_error
        return wrapper
    return decorator

class RateLimiter:
    """Rate limiter for API calls"""
    def __init__(self, calls: int, period: int):
        self.calls = calls
        self.period = period
        self.timestamps = []

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            self.timestamps = [ts for ts in self.timestamps if now - ts < self.period]

            if len(self.timestamps) >= self.calls:
                sleep_time = self.timestamps[0] + self.period - now
                if sleep_time > 0:
                    logger.info(f"Rate limit reached. Waiting {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                    self.timestamps.pop(0)

            self.timestamps.append(now)
            return func(*args, **kwargs)
        return wrapper

def validate_auth(func):
    """Validate authentication before API calls"""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, 'credentials') or not all(self.credentials.values()):
            raise AuthenticationError("Missing or invalid credentials")
        return func(self, *args, **kwargs)
    return wrapper
