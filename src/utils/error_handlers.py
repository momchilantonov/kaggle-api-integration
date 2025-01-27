from typing import Optional, Type, Any
import logging
import functools
import time
from requests.exceptions import RequestException, HTTPError
import click  # Adding click for CLI-friendly error messages

__all__ = ["retry_with_backoff", "handle_api_errors", "RateLimiter", "CLIError"]

logger = logging.getLogger(__name__)

class KaggleAPIError(Exception):
    """Base exception for Kaggle API errors"""
    def __init__(self, message: str, status_code: Optional[int] = None, response: Any = None):
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(self.message)

    def get_cli_message(self) -> str:
        """Return a CLI-friendly error message"""
        if self.status_code:
            return f"API Error ({self.status_code}): {self.message}"
        return f"API Error: {self.message}"

class AuthenticationError(KaggleAPIError):
    """Authentication failed"""
    def get_cli_message(self) -> str:
        return "Authentication failed. Please check your Kaggle credentials."

class RateLimitError(KaggleAPIError):
    """Rate limit exceeded"""
    def get_cli_message(self) -> str:
        return "Rate limit exceeded. Please wait before making more requests."

class ResourceNotFoundError(KaggleAPIError):
    """Requested resource not found"""
    def get_cli_message(self) -> str:
        return f"Resource not found: {self.message}"

class ValidationError(KaggleAPIError):
    """Invalid request parameters"""
    def get_cli_message(self) -> str:
        return f"Validation error: {self.message}"

class CLIError(Exception):
    """Custom exception for CLI-specific errors"""
    def __init__(self, message: str, exit_code: int = 1):
        self.message = message
        self.exit_code = exit_code
        super().__init__(self.message)

def handle_api_errors(func):
    """Decorator to handle API errors and present them in CLI-friendly format"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except HTTPError as e:
            status_code = e.response.status_code
            if status_code == 401:
                error = AuthenticationError("Authentication failed", status_code, e.response)
                click.echo(click.style(error.get_cli_message(), fg='red'), err=True)
                raise click.Abort()
            elif status_code == 403:
                error = RateLimitError("Rate limit exceeded", status_code, e.response)
                click.echo(click.style(error.get_cli_message(), fg='yellow'), err=True)
                raise click.Abort()
            elif status_code == 404:
                error = ResourceNotFoundError("Resource not found", status_code, e.response)
                click.echo(click.style(error.get_cli_message(), fg='red'), err=True)
                raise click.Abort()
            elif status_code == 422:
                error = ValidationError("Invalid request", status_code, e.response)
                click.echo(click.style(error.get_cli_message(), fg='red'), err=True)
                raise click.Abort()
            error = KaggleAPIError(f"API error: {str(e)}", status_code, e.response)
            click.echo(click.style(error.get_cli_message(), fg='red'), err=True)
            raise click.Abort()
        except RequestException as e:
            click.echo(click.style(f"Request failed: {str(e)}", fg='red'), err=True)
            raise click.Abort()
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            click.echo(click.style(f"An unexpected error occurred: {str(e)}", fg='red'), err=True)
            raise click.Abort()

    return wrapper

def retry_with_backoff(
    max_retries: int = 3,
    backoff_factor: float = 1.5,
    error_types: tuple = (RequestException,)
):
    """Retry decorator with exponential backoff and CLI feedback"""
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
                        if isinstance(e, HTTPError):
                            handle_api_errors(lambda: None)()
                        click.echo(click.style(f"Failed after {max_retries} attempts: {str(e)}", fg='red'), err=True)
                        raise click.Abort()
                    wait_time = backoff_factor * (2 ** attempt)
                    click.echo(click.style(
                        f"Attempt {attempt + 1} failed. Retrying in {wait_time:.1f} seconds...",
                        fg='yellow'
                    ), err=True)
                    time.sleep(wait_time)
            raise last_error
        return wrapper
    return decorator

class RateLimiter:
    """Rate limiter for API calls with CLI feedback"""
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
                    click.echo(click.style(
                        f"Rate limit reached. Waiting {sleep_time:.1f} seconds...",
                        fg='yellow'
                    ), err=True)
                    time.sleep(sleep_time)
                    self.timestamps.pop(0)

            self.timestamps.append(now)
            return func(*args, **kwargs)
        return wrapper
