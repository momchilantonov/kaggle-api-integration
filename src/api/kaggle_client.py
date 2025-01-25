import time
from typing import Dict, Optional, Any, Union
import requests
from pathlib import Path

from config.settings import (
    validate_environment,
    API_ENDPOINTS,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    RETRY_DELAY,
    setup_logger
)

logger = setup_logger('kaggle_client', 'kaggle_client.log')

class KaggleAPIClient:
    """Base client for interacting with Kaggle API"""

    def __init__(self):
        """Initialize the Kaggle API client with credentials"""
        self.credentials = validate_environment()
        self.auth = (self.credentials['username'], self.credentials['key'])
        self.session = requests.Session()
        self.session.auth = self.auth
        logger.info("Kaggle API client initialized")

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        files: Optional[Dict] = None,
        stream: bool = False,
        json: Optional[Dict] = None
    ) -> requests.Response:
        """
        Make a request to the Kaggle API with retry logic

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint to call
            params: Query parameters
            data: Form data
            files: Files to upload
            stream: Whether to stream the response
            json: JSON data for the request body

        Returns:
            Response from the API

        Raises:
            requests.exceptions.RequestException: If the request fails after all retries
            ValueError: If the endpoint is not found in API_ENDPOINTS
        """
        if endpoint not in API_ENDPOINTS:
            raise ValueError(f"Unknown endpoint: {endpoint}")

        url = API_ENDPOINTS[endpoint]
        retry_count = 0

        while retry_count < MAX_RETRIES:
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    data=data,
                    files=files,
                    json=json,
                    stream=stream,
                    timeout=REQUEST_TIMEOUT
                )
                response.raise_for_status()
                return response

            except requests.exceptions.RequestException as e:
                retry_count += 1
                if retry_count == MAX_RETRIES:
                    logger.error(f"Request failed after {MAX_RETRIES} retries: {str(e)}")
                    raise
                logger.warning(
                    f"Request failed, attempt {retry_count} of {MAX_RETRIES}: {str(e)}"
                )
                time.sleep(RETRY_DELAY * retry_count)  # Exponential backoff

    def get(self, endpoint: str, **kwargs) -> requests.Response:
        """Make a GET request"""
        return self._make_request('GET', endpoint, **kwargs)

    def post(self, endpoint: str, **kwargs) -> requests.Response:
        """Make a POST request"""
        return self._make_request('POST', endpoint, **kwargs)

    def put(self, endpoint: str, **kwargs) -> requests.Response:
        """Make a PUT request"""
        return self._make_request('PUT', endpoint, **kwargs)

    def delete(self, endpoint: str, **kwargs) -> requests.Response:
        """Make a DELETE request"""
        return self._make_request('DELETE', endpoint, **kwargs)

    def download_file(
        self,
        response: requests.Response,
        path: Union[str, Path],
        filename: str
    ) -> Path:
        """
        Download a file from a response to a specified path

        Args:
            response: Response object from a request
            path: Directory to save the file
            filename: Name of the file to save

        Returns:
            Path to the downloaded file
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / filename

        with open(file_path, 'wb') as f:
            if response.headers.get('content-length'):
                total_size = int(response.headers['content-length'])
                chunk_size = 8192
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
            else:
                f.write(response.content)

        logger.info(f"Downloaded file to {file_path}")
        return file_path

    def upload_file(
        self,
        endpoint: str,
        file_path: Union[str, Path],
        additional_params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Upload a file to Kaggle

        Args:
            endpoint: API endpoint for the upload
            file_path: Path to the file to upload
            additional_params: Additional parameters for the upload

        Returns:
            Response from the API
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'rb') as f:
            files = {'file': (file_path.name, f)}
            response = self.post(
                endpoint,
                files=files,
                data=additional_params
            )

        return response.json()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.session.close()
