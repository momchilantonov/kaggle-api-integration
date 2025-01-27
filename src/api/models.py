from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

from src.utils.error_handlers import handle_api_errors, validate_auth
from .kaggle_client import KaggleAPIClient

@dataclass
class ModelMetadata:
    name: str
    version_name: str
    description: str
    framework: str
    task_ids: List[str]
    training_data: Optional[str] = None
    model_type: Optional[str] = None
    training_params: Optional[Dict] = None
    license: Optional[str] = None

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}

class ModelClient:
    def __init__(self, client: KaggleAPIClient):
        self.client = client

    @handle_api_errors
    @validate_auth
    def list_models(
        self,
        owner: Optional[str] = None,
        search: Optional[str] = None,
        page: int = 1,
        **kwargs
    ) -> List[Dict]:
        params = {'owner': owner, 'search': search, 'page': page, **kwargs}
        response = self.client.get('models/list', params=params)
        return response.json()

    @handle_api_errors
    @validate_auth
    def pull_model(
        self,
        owner: str,
        model_name: str,
        version: Optional[str] = None,
        path: Optional[Union[str, Path]] = None
    ) -> Path:
        path = Path(path) if path else Path.cwd()
        path.mkdir(parents=True, exist_ok=True)

        params = {
            'ownerSlug': owner,
            'modelSlug': model_name,
            'versionNumber': version
        }
        response = self.client.get('models/pull', params=params, stream=True)
        return self.client.download_file(response.url, path / f"{model_name}.zip")

    @handle_api_errors
    @validate_auth
    def push_model(
        self,
        path: Union[str, Path],
        metadata: ModelMetadata,
        version_notes: Optional[str] = None
    ) -> Dict:
        # Initialize model
        result = self.client.post(
            'models/initiate',
            json=metadata.to_dict()
        ).json()

        # Upload files
        path = Path(path)
        for file_path in path.glob('**/*'):
            if file_path.is_file():
                self.client.post(
                    'models/upload',
                    files={'file': open(file_path, 'rb')},
                    data={
                        'path': str(file_path.relative_to(path)),
                        'modelSlug': metadata.name
                    }
                )

        # Push model
        return self.client.post(
            'models/push',
            json={
                'modelSlug': metadata.name,
                'versionNotes': version_notes
            }
        ).json()

    @handle_api_errors
    @validate_auth
    def get_model_status(
        self,
        owner: str,
        model_name: str,
        version: Optional[str] = None
    ) -> Dict:
        params = {
            'ownerSlug': owner,
            'modelSlug': model_name,
            'versionNumber': version
        }
        return self.client.get('models/status', params=params).json()

    def wait_for_model_ready(
        self,
        owner: str,
        model_name: str,
        timeout: int = 3600,
        check_interval: int = 10
    ) -> Dict:
        import time
        start_time = time.time()

        while True:
            status = self.get_model_status(owner, model_name)

            if status.get('status') == 'complete':
                return status

            if status.get('status') == 'error':
                raise RuntimeError(f"Model failed: {status.get('errorMessage')}")

            if time.time() - start_time > timeout:
                raise TimeoutError(f"Model not ready after {timeout} seconds")

            time.sleep(check_interval)
