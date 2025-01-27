from pathlib import Path
from typing import Dict, Optional, Union

from src.utils.error_handlers import handle_api_errors, validate_auth
from .kaggle_client import KaggleAPIClient

class FileClient:
    def __init__(self, client: KaggleAPIClient):
        self.client = client

    @handle_api_errors
    @validate_auth
    def get_file(
        self,
        dataset_owner: str,
        dataset_name: str,
        file_name: str,
        path: Optional[Union[str, Path]] = None,
        force: bool = False
    ) -> Path:
        path = Path(path) if path else Path.cwd()
        file_path = path / file_name

        if file_path.exists() and not force:
            return file_path

        response = self.client.get(
            'files/get',
            params={
                'datasetOwner': dataset_owner,
                'datasetName': dataset_name,
                'fileName': file_name
            },
            stream=True
        )

        return self.client.download_file(response.url, file_path)

    @handle_api_errors
    @validate_auth
    def upload_file(
        self,
        dataset_owner: str,
        dataset_name: str,
        file_path: Union[str, Path],
        target_path: Optional[str] = None
    ) -> Dict:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        return self.client.post(
            'files/upload',
            files={'file': open(file_path, 'rb')},
            data={
                'datasetOwner': dataset_owner,
                'datasetName': dataset_name,
                'path': target_path or file_path.name
            }
        ).json()

    @handle_api_errors
    @validate_auth
    def delete_file(
        self,
        dataset_owner: str,
        dataset_name: str,
        file_name: str
    ) -> Dict:
        return self.client.delete(
            'files/delete',
            params={
                'datasetOwner': dataset_owner,
                'datasetName': dataset_name,
                'fileName': file_name
            }
        ).json()
