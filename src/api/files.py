from typing import Dict, Optional, Union
from pathlib import Path

from .kaggle_client import KaggleAPIClient
from config.settings import setup_logger

logger = setup_logger('kaggle_files', 'kaggle_files.log')

class FileClient:
    """Client for handling Kaggle file operations"""

    def __init__(self, client: KaggleAPIClient):
        """Initialize with a KaggleAPIClient instance"""
        self.client = client

    def get_file(
        self,
        dataset_owner: str,
        dataset_name: str,
        file_name: str,
        path: Optional[Union[str, Path]] = None,
        force: bool = False
    ) -> Path:
        """
        Get a specific file from a dataset

        Args:
            dataset_owner: Owner of the dataset
            dataset_name: Name of the dataset
            file_name: Name of the file to get
            path: Path to save the file (default: current directory)
            force: Whether to overwrite existing file

        Returns:
            Path to the downloaded file
        """
        path = Path(path) if path else Path.cwd()
        path.mkdir(parents=True, exist_ok=True)

        file_path = path / file_name
        if file_path.exists() and not force:
            logger.info(f"File {file_path} already exists. Use force=True to overwrite.")
            return file_path

        response = self.client.get(
            'files_get',
            params={
                'datasetOwner': dataset_owner,
                'datasetName': dataset_name,
                'fileName': file_name
            },
            stream=True
        )

        return self.client.download_file(response, path, file_name)

    def upload_file(
        self,
        dataset_owner: str,
        dataset_name: str,
        file_path: Union[str, Path],
        target_path: Optional[str] = None
    ) -> Dict:
        """
        Upload a file to a dataset

        Args:
            dataset_owner: Owner of the dataset
            dataset_name: Name of the dataset
            file_path: Path to the file to upload
            target_path: Target path in the dataset (default: file name)

        Returns:
            Response from the API
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        target_path = target_path or file_path.name

        additional_params = {
            'datasetOwner': dataset_owner,
            'datasetName': dataset_name,
            'path': target_path
        }

        return self.client.upload_file(
            'files_upload',
            file_path,
            additional_params
        )

    def delete_file(
        self,
        dataset_owner: str,
        dataset_name: str,
        file_name: str
    ) -> Dict:
        """
        Delete a file from a dataset

        Args:
            dataset_owner: Owner of the dataset
            dataset_name: Name of the dataset
            file_name: Name of the file to delete

        Returns:
            Response from the API
        """
        response = self.client.delete(
            'files_delete',
            params={
                'datasetOwner': dataset_owner,
                'datasetName': dataset_name,
                'fileName': file_name
            }
        )

        result = response.json()
        logger.info(f"Deleted file {file_name} from {dataset_owner}/{dataset_name}")
        return result

    def verify_file_hash(self, file_path: Union[str, Path], expected_hash: str) -> bool:
        """
        Verify file hash to ensure integrity

        Args:
            file_path: Path to the file
            expected_hash: Expected SHA-256 hash

        Returns:
            True if hash matches, False otherwise
        """
        import hashlib

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        actual_hash = sha256_hash.hexdigest()

        matches = actual_hash.lower() == expected_hash.lower()
        if not matches:
            logger.warning(
                f"Hash mismatch for {file_path}. "
                f"Expected: {expected_hash}, Got: {actual_hash}"
            )

        return matches

    def get_file_size(self, file_path: Union[str, Path]) -> int:
        """
        Get file size in bytes

        Args:
            file_path: Path to the file

        Returns:
            Size of the file in bytes
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        return file_path.stat().st_size
