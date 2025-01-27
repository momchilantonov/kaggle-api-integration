import requests
import logging
from typing import Optional, Dict, Any
from .error_handlers import retry_with_backoff, handle_api_errors, RateLimiter

logger = logging.getLogger(__name__)

class RequestManager:
    def __init__(self, base_url: str, auth: tuple, rate_limit: int = 10):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.auth = auth
        self._rate_limiter = RateLimiter(calls=rate_limit, period=60)

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
        timeout: int = 30
    ) -> requests.Response:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = self.session.request(
            method=method,
            url=url,
            params=params,
            data=data,
            files=files,
            stream=stream,
            timeout=timeout
        )
        response.raise_for_status()
        return response

    def get(self, endpoint: str, **kwargs) -> Any:
        return self.make_request('GET', endpoint, **kwargs)

    def post(self, endpoint: str, **kwargs) -> Any:
        return self.make_request('POST', endpoint, **kwargs)

    def put(self, endpoint: str, **kwargs) -> Any:
        return self.make_request('PUT', endpoint, **kwargs)

    def delete(self, endpoint: str, **kwargs) -> Any:
        return self.make_request('DELETE', endpoint, **kwargs)

    def download_file(self, url: str, path: str) -> None:
        with self.get(url, stream=True) as r:
            r.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
