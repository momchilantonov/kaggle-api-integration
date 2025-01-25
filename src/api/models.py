from typing import Dict, List, Optional, Union
from pathlib import Path
import time
from dataclasses import dataclass

from .kaggle_client import KaggleAPIClient
from config.settings import setup_logger

logger = setup_logger('kaggle_models', 'kaggle_models.log')

@dataclass
class ModelMetadata:
    """Model metadata for creation/update operations"""
    name: str
    version_name: str
    description: str
    framework: str  # e.g., 'PyTorch', 'TensorFlow', etc.
    task_ids: List[str]  # e.g., ['computer-vision', 'image-classification']
    training_data: Optional[str] = None
    model_type: Optional[str] = None
    training_params: Optional[Dict] = None
    license: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert metadata to dictionary format"""
        return {k: v for k, v in self.__dict__.items() if v is not None}

class ModelClient:
    """Client for handling Kaggle model operations"""

    def __init__(self, client: KaggleAPIClient):
        """Initialize with a KaggleAPIClient instance"""
        self.client = client

    def list_models(
        self,
        owner: Optional[str] = None,
        search: Optional[str] = None,
        framework: Optional[str] = None,
        task: Optional[str] = None,
        sort_by: Optional[str] = None,
        page: int = 1,
        page_size: int = 20
    ) -> List[Dict]:
        """
        List models on Kaggle with filtering options

        Args:
            owner: Filter by model owner
            search: Search terms
            framework: Filter by framework (PyTorch, TensorFlow, etc.)
            task: Filter by task type
            sort_by: Sort results (latest, popular, etc.)
            page: Page number for pagination
            page_size: Number of results per page

        Returns:
            List of models matching criteria
        """
        params = {
            'owner': owner,
            'search': search,
            'framework': framework,
            'task': task,
            'sortBy': sort_by,
            'page': page,
            'pageSize': page_size
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        response = self.client.get('models_list', params=params)
        models = response.json()
        logger.info(f"Found {len(models)} models matching criteria")
        return models

    def pull_model(
        self,
        owner: str,
        model_name: str,
        version: Optional[str] = None,
        path: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Download a model

        Args:
            owner: Model owner's username
            model_name: Name of the model
            version: Specific version to pull (latest if None)
            path: Path to save the model

        Returns:
            Path to the downloaded model
        """
        path = Path(path) if path else Path.cwd()
        path.mkdir(parents=True, exist_ok=True)

        params = {
            'ownerSlug': owner,
            'modelSlug': model_name
        }
        if version:
            params['versionNumber'] = version

        response = self.client.get(
            'model_pull',
            params=params,
            stream=True
        )

        model_path = path / f"{model_name}.zip"
        model_path = self.client.download_file(response, path, model_path.name)

        return model_path

    def push_model(
        self,
        path: Union[str, Path],
        metadata: ModelMetadata,
        public: bool = True
    ) -> Dict:
        """
        Push a model to Kaggle

        Args:
            path: Path to the model files
            metadata: Model metadata
            public: Whether the model should be public

        Returns:
            Response from the API
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if not path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {path}")

        # Initialize model upload
        init_response = self.client.post(
            'model_initiate',
            json={
                **metadata.to_dict(),
                'isPublic': public
            }
        )

        # Upload model files
        for file_path in path.rglob('*'):
            if file_path.is_file():
                relative_path = str(file_path.relative_to(path))
                self.client.upload_file(
                    'model_upload',
                    file_path,
                    {
                        'path': relative_path,
                        'modelSlug': metadata.name
                    }
                )

        # Push model
        push_response = self.client.post(
            'model_push',
            json={'modelSlug': metadata.name}
        )

        return push_response.json()

    def get_model_status(
        self,
        owner: str,
        model_name: str,
        version: Optional[str] = None
    ) -> Dict:
        """
        Get model status

        Args:
            owner: Model owner's username
            model_name: Name of the model
            version: Specific version to check

        Returns:
            Model status information
        """
        params = {
            'ownerSlug': owner,
            'modelSlug': model_name
        }
        if version:
            params['versionNumber'] = version

        response = self.client.get('model_status', params=params)
        return response.json()

    def wait_for_model_ready(
        self,
        owner: str,
        model_name: str,
        version: Optional[str] = None,
        timeout: int = 3600,  # 1 hour
        check_interval: int = 10
    ) -> Dict:
        """
        Wait for model to be ready

        Args:
            owner: Model owner's username
            model_name: Name of the model
            version: Specific version to check
            timeout: Maximum time to wait in seconds
            check_interval: Time between checks in seconds

        Returns:
            Final model status

        Raises:
            TimeoutError: If model is not ready within timeout
        """
        start_time = time.time()

        while True:
            status = self.get_model_status(owner, model_name, version)

            if status.get('status') == 'complete':
                logger.info(f"Model {owner}/{model_name} is ready")
                return status

            if status.get('status') == 'error':
                error_msg = status.get('errorMessage', 'Unknown error')
                logger.error(f"Model failed: {error_msg}")
                raise RuntimeError(f"Model failed: {error_msg}")

            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Model not ready after {timeout} seconds"
                )

            time.sleep(check_interval)
