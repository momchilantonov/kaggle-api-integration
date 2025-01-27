from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

from src.utils.error_handlers import handle_api_errors, validate_auth
from .kaggle_client import KaggleAPIClient

@dataclass
class DatasetMetadata:
    title: str
    slug: str
    description: str
    licenses: List[Dict[str, str]]
    keywords: List[str]
    collaborators: Optional[List[str]] = None

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}

class DatasetClient:
    def __init__(self, client: KaggleAPIClient):
        self.client = client

    @handle_api_errors
    @validate_auth
    def list_datasets(
        self,
        search: Optional[str] = None,
        user: Optional[str] = None,
        page: int = 1,
        **kwargs
    ) -> List[Dict]:
        params = {'search': search, 'user': user, 'page': page, **kwargs}
        response = self.client.get('datasets/list', params=params)
        return response.json()

    @handle_api_errors
    @validate_auth
    def download_dataset(
        self,
        owner_slug: str,
        dataset_slug: str,
        path: Optional[Union[str, Path]] = None,
        unzip: bool = True
    ) -> Path:
        path = Path(path) if path else Path.cwd()
        path.mkdir(parents=True, exist_ok=True)

        response = self.client.get(
            'datasets/download',
            params={'datasetSlug': f"{owner_slug}/{dataset_slug}"},
            stream=True
        )

        file_path = self.client.download_file(
            response.url,
            path / f"{dataset_slug}.zip"
        )

        if unzip:
            import zipfile
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(path)
            file_path.unlink()
            return path / dataset_slug

        return file_path

    @handle_api_errors
    @validate_auth
    def create_dataset(
        self,
        folder_path: Union[str, Path],
        metadata: DatasetMetadata,
        public: bool = True
    ) -> Dict:
        result = self.client.post(
            'datasets/create',
            json={
                **metadata.to_dict(),
                'isPublic': public
            }
        )

        self._upload_files(folder_path)
        return result.json()

    def _upload_files(self, folder_path: Union[str, Path]) -> None:
        folder_path = Path(folder_path)
        for file_path in folder_path.glob('**/*'):
            if file_path.is_file():
                self.client.post(
                    'datasets/upload/file',
                    files={'file': open(file_path, 'rb')},
                    data={'path': str(file_path.relative_to(folder_path))}
                )
