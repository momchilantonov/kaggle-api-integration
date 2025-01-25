from typing import Dict, List, Optional, Union
from pathlib import Path
import json
from dataclasses import dataclass

from .kaggle_client import KaggleAPIClient
from config.settings import setup_logger

logger = setup_logger('kaggle_datasets', 'kaggle_datasets.log')

@dataclass
class DatasetMetadata:
    """Dataset metadata for creation/update operations"""
    title: str
    slug: str
    description: str
    licenses: List[Dict[str, str]]
    keywords: List[str]
    collaborators: List[str] = None
    data_resources: List[Dict[str, str]] = None

    def to_dict(self) -> Dict:
        """Convert metadata to dictionary format"""
        return {k: v for k, v in self.__dict__.items() if v is not None}

class DatasetClient:
    """Client for interacting with Kaggle Datasets API"""

    def __init__(self, client: KaggleAPIClient):
        """Initialize with a KaggleAPIClient instance"""
        self.client = client

    def list_datasets(
        self,
        search: Optional[str] = None,
        user: Optional[str] = None,
        file_type: Optional[str] = None,
        license_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        sort_by: Optional[str] = None,
        size: Optional[str] = None,
        max_size: Optional[int] = None,
        min_size: Optional[int] = None,
        page: int = 1,
        page_size: int = 20
    ) -> List[Dict]:
        """
        List datasets on Kaggle with filtering options

        Args:
            search: Search terms
            user: Filter by user
            file_type: Filter by file type
            license_name: Filter by license
            tags: Filter by tags
            sort_by: Sort results (options: hotness, votes, updated, active)
            size: Filter by size category
            max_size: Maximum size in bytes
            min_size: Minimum size in bytes
            page: Page number for pagination
            page_size: Number of results per page

        Returns:
            List of datasets matching criteria
        """
        params = {
            'search': search,
            'user': user,
            'fileType': file_type,
            'licenseName': license_name,
            'tags': ','.join(tags) if tags else None,
            'sortBy': sort_by,
            'size': size,
            'maxSize': max_size,
            'minSize': min_size,
            'page': page,
            'pageSize': page_size
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        response = self.client.get('datasets_list', params=params)
        datasets = response.json()
        logger.info(f"Found {len(datasets)} datasets matching criteria")
        return datasets

    def download_dataset(
        self,
        owner_slug: str,
        dataset_slug: str,
        file_name: Optional[str] = None,
        path: Optional[Union[str, Path]] = None,
        unzip: bool = True
    ) -> Path:
        """
        Download a dataset or specific file

        Args:
            owner_slug: Dataset owner's username
            dataset_slug: Dataset name
            file_name: Specific file to download (downloads all if None)
            path: Path to save the dataset
            unzip: Whether to unzip the downloaded file

        Returns:
            Path to the downloaded file(s)
        """
        path = Path(path) if path else Path.cwd()
        path.mkdir(parents=True, exist_ok=True)

        params = {
            'datasetSlug': f"{owner_slug}/{dataset_slug}"
        }
        if file_name:
            params['fileName'] = file_name

        response = self.client.get(
            'dataset_download',
            params=params,
            stream=True
        )

        if file_name:
            return self.client.download_file(response, path, file_name)
        else:
            zip_path = self.client.download_file(
                response,
                path,
                f"{dataset_slug}.zip"
            )

            if unzip:
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(path / dataset_slug)
                zip_path.unlink()  # Remove zip file after extraction
                return path / dataset_slug

            return zip_path

    def create_dataset(
        self,
        folder_path: Union[str, Path],
        metadata: DatasetMetadata,
        public: bool = True
    ) -> Dict:
        """
        Create a new dataset

        Args:
            folder_path: Path to folder containing dataset files
            metadata: Dataset metadata
            public: Whether the dataset should be public

        Returns:
            Response from the API
        """
        folder_path = Path(folder_path)
        if not folder_path.is_dir():
            raise NotADirectoryError(f"{folder_path} is not a directory")

        # Create initial dataset
        create_response = self.client.post(
            'dataset_create',
            json={
                'title': metadata.title,
                'slug': metadata.slug,
                'owner_slug': self.client.credentials['username'],
                'description': metadata.description,
                'licenses': metadata.licenses,
                'keywords': metadata.keywords,
                'collaborators': metadata.collaborators,
                'isPublic': public
            }
        )

        # Upload each file in the directory
        for file_path in folder_path.glob('**/*'):
            if file_path.is_file():
                self.client.upload_file(
                    'dataset_upload_file',
                    file_path,
                    {
                        'path': str(file_path.relative_to(folder_path))
                    }
                )

        return create_response.json()

    def create_version(
        self,
        folder_path: Union[str, Path],
        version_notes: str,
        delete_old_versions: bool = False
    ) -> Dict:
        """
        Create a new version of an existing dataset

        Args:
            folder_path: Path to folder containing updated files
            version_notes: Notes describing the changes
            delete_old_versions: Whether to delete previous versions

        Returns:
            Response from the API
        """
        folder_path = Path(folder_path)
        if not folder_path.is_dir():
            raise NotADirectoryError(f"{folder_path} is not a directory")

        # Upload new version
        response = self.client.post(
            'dataset_create_version',
            json={
                'versionNotes': version_notes,
                'deleteOldVersions': delete_old_versions
            }
        )

        # Upload updated files
        for file_path in folder_path.glob('**/*'):
            if file_path.is_file():
                self.client.upload_file(
                    'dataset_upload_file',
                    file_path,
                    {
                        'path': str(file_path.relative_to(folder_path))
                    }
                )

        return response.json()

    def update_metadata(
        self,
        owner_slug: str,
        dataset_slug: str,
        metadata: DatasetMetadata
    ) -> Dict:
        """
        Update dataset metadata

        Args:
            owner_slug: Dataset owner's username
            dataset_slug: Dataset name
            metadata: Updated metadata

        Returns:
            Response from the API
        """
        response = self.client.post(
            'dataset_metadata',
            json={
                'ownerSlug': owner_slug,
                'datasetSlug': dataset_slug,
                **metadata.to_dict()
            }
        )
        return response.json()

    def delete_dataset(
        self,
        owner_slug: str,
        dataset_slug: str
    ) -> Dict:
        """
        Delete a dataset

        Args:
            owner_slug: Dataset owner's username
            dataset_slug: Dataset name

        Returns:
            Response from the API
        """
        response = self.client.delete(
            'dataset_delete',
            params={
                'ownerSlug': owner_slug,
                'datasetSlug': dataset_slug
            }
        )
        return response.json()
