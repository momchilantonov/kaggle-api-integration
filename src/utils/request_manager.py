import requests
import logging
from typing import Optional, Dict, Any
from .error_handlers import retry_with_backoff, handle_api_errors, RateLimiter, RateLimitError
import click

logger = logging.getLogger(__name__)

class RequestManager:
    def __init__(self, base_url: str, auth: tuple, rate_limit: int = 10, rate_period: int = 60):
        """
        Initialize RequestManager with rate limiting

        Args:
            base_url: Base URL for API requests
            auth: Authentication tuple (username, key)
            rate_limit: Maximum number of requests per period
            rate_period: Time period in seconds for rate limiting
        """
        self.base_url = base_url
        self.session = requests.Session()
        self.session.auth = auth
        self._rate_limiter = RateLimiter(calls=rate_limit, period=rate_period)

        # Track request counts
        self.request_count = 0
        self.rate_limit = rate_limit

        # Apply rate limiting to HTTP methods
        self.get = self._rate_limiter(self.get)
        self.post = self._rate_limiter(self.post)
        self.put = self._rate_limiter(self.put)
        self.delete = self._rate_limiter(self.delete)
        self.download_file = self._rate_limiter(self.download_file)

    @retry_with_backoff(max_retries=3)
    @handle_api_errors
    def make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        files: Optional[Dict] = None,
        stream: bool = False,
        timeout: int = 30,
        json: Optional[Dict] = None
    ) -> requests.Response:
        """Make a rate-limited API request"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        # Check rate limit before making request
        self.request_count += 1
        remaining = self.rate_limit - self.request_count

        if remaining < (self.rate_limit * 0.2):  # Warning at 20% remaining
            click.echo(click.style(
                f"\nWarning: {remaining} requests remaining in current period",
                fg='yellow'
            ), err=True)

        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                data=data,
                files=files,
                json=json,
                stream=stream,
                timeout=timeout
            )
            response.raise_for_status()

            # Update rate limit info from response headers if available
            if 'X-RateLimit-Remaining' in response.headers:
                remaining = int(response.headers['X-RateLimit-Remaining'])
                if remaining < (self.rate_limit * 0.2):
                    click.echo(click.style(
                        f"\nAPI Rate Limit Warning: {remaining} requests remaining",
                        fg='yellow'
                    ), err=True)

            return response

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limit exceeded
                reset_time = e.response.headers.get('X-RateLimit-Reset', 'unknown')
                raise RateLimitError(
                    f"Rate limit exceeded. Reset at: {reset_time}",
                    status_code=429,
                    response=e.response
                )
            raise

    def get(self, endpoint: str, **kwargs) -> Any:
        """Make GET request"""
        return self.make_request('GET', endpoint, **kwargs)

    def post(self, endpoint: str, **kwargs) -> Any:
        """Make POST request"""
        return self.make_request('POST', endpoint, **kwargs)

    def put(self, endpoint: str, **kwargs) -> Any:
        """Make PUT request"""
        return self.make_request('PUT', endpoint, **kwargs)

    def delete(self, endpoint: str, **kwargs) -> Any:
        """Make DELETE request"""
        return self.make_request('DELETE', endpoint, **kwargs)

    def download_file(self, url: str, path: str) -> None:
        """Download file with rate limiting and progress bar"""
        with self.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))

            with open(path, 'wb') as f:
                with click.progressbar(
                    length=total,
                    label='Downloading file'
                ) as bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))

    def close(self):
        """Close the session"""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
