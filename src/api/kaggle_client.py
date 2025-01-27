from pathlib import Path
from typing import Dict, Optional, Union
import logging

from src.utils.request_manager import RequestManager
from src.utils.auth_validator import AuthValidator
from src.utils.error_handlers import handle_api_errors, validate_auth

logger = logging.getLogger(__name__)

class KaggleAPIClient:
    API_BASE_URL = "https://www.kaggle.com/api/v1"

    def __init__(self, credentials: Optional[Dict] = None):
        self.credentials = AuthValidator.validate_credentials(credentials)
        self.request_manager = RequestManager(
            base_url=self.API_BASE_URL,
            auth=(self.credentials['username'], self.credentials['key'])
        )

    @validate_auth
    def get(self, endpoint: str, **kwargs):
        return self.request_manager.get(endpoint, **kwargs)

    @validate_auth
    def post(self, endpoint: str, **kwargs):
        return self.request_manager.post(endpoint, **kwargs)

    @validate_auth
    def put(self, endpoint: str, **kwargs):
        return self.request_manager.put(endpoint, **kwargs)

    @validate_auth
    def delete(self, endpoint: str, **kwargs):
        return self.request_manager.delete(endpoint, **kwargs)

    @validate_auth
    @handle_api_errors
    def download_file(self, url: str, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.request_manager.download_file(url, str(path))
        return path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.request_manager.close()
