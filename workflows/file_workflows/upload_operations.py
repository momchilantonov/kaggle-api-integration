from pathlib import Path
import yaml
from typing import Optional, List, Dict, Union
import logging
import shutil
from datetime import datetime
import json

from src.api.kaggle_client import KaggleAPIClient
from src.api.files import FileClient
from src.handlers.data_handlers import DataHandler
from src.utils.helpers import timer, retry_on_exception, compress_file

logger = logging.getLogger(__name__)

class FileUploadManager:
    def __init__(self):
        """Initialize the file upload manager"""
        self.kaggle_client = KaggleAPIClient()
        self.file_client = FileClient(self.kaggle_client)
        self.data_handler = DataHandler()
        self._load_configs()

    def _load_configs(self):
        """Load operational configurations"""
        try:
            with open('operational_configs/file_configs/file_operations.yaml', 'r') as f:
                self.file_config = yaml.safe_load(f)
            with open('operational_configs/file_configs/file_paths.yaml', 'r') as f:
                self.path_config = yaml.safe_load(f)
            logger.info("Successfully loaded file configurations")
        except Exception as e:
            logger.error(f"Error loading configurations: {str(e)}")
            raise

    @timer
    @retry_on_exception(retries=3, delay=1)
    def upload_file(
        self,
        file_path: Union[str, Path],
        dataset_owner: str,
        dataset_name: str,
        target_path: Optional[str] = None
    ) -> Dict:
        """
        Upload a file to a dataset

        Args:
            file_path: Path to the file
            dataset_owner: Owner of the dataset
            dataset_name: Name of the dataset
            target_path: Target path in dataset

        Returns:
            Upload response
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Check file size and compress if needed
            if self._should_compress(file_path):
                file_path = self._compress_file(file_path)

            # Determine target path
            if target_path is None:
                target_path = file_path.name

            result = self.file_client.upload_file(
                dataset_owner=dataset_owner,
                dataset_name=dataset_name,
                file_path=file_path,
                target_path=target_path
            )

            logger.info(f"Successfully uploaded {file_path.name} to {dataset_name}")
            return result

        except Exception as e:
            logger.error(f"Error uploading file: {str(e)}")
            raise

    def _should_compress(self, file_path: Path) -> bool:
        """Determine if file should be compressed"""
        try:
            max_size = self.file_config['file_settings']['uploads']['max_file_size']
            return file_path.stat().st_size > max_size

        except Exception as e:
            logger.error(f"Error checking file size: {str(e)}")
            return False

    def _compress_file(self, file_path: Path) -> Path:
        """Compress file before upload"""
        try:
            compression_type = self.file_config['file_settings']['compression_type']
            compressed_path = compress_file(
                file_path,
                method=compression_type
            )
            logger.info(f"Compressed {file_path.name}")
            return compressed_path

        except Exception as e:
            logger.error(f"Error compressing file: {str(e)}")
            raise

    @timer
    def batch_upload_files(
        self,
        file_paths: List[Union[str, Path]],
        dataset_owner: str,
        dataset_name: str,
        target_directory: Optional[str] = None
    ) -> List[Dict]:
        """
        Upload multiple files in batch

        Args:
            file_paths: List of file paths
            dataset_owner: Owner of the dataset
            dataset_name: Name of the dataset
            target_directory: Target directory in dataset

        Returns:
            List of upload responses
        """
        try:
            results = []
            for file_path in file_paths:
                file_path = Path(file_path)
                target_path = f"{target_directory}/{file_path.name}" if target_directory else None

                try:
                    result = self.upload_file(
                        file_path,
                        dataset_owner,
                        dataset_name,
                        target_path
                    )
                    results.append({
                        'file': file_path.name,
                        'status': 'success',
                        'result': result
                    })
                except Exception as e:
                    results.append({
                        'file': file_path.name,
                        'status': 'error',
                        'error': str(e)
                    })

            return results

        except Exception as e:
            logger.error(f"Error in batch upload: {str(e)}")
            raise

    @timer
    def prepare_files_for_upload(
        self,
        source_dir: Union[str, Path],
        file_patterns: Optional[List[str]] = None
    ) -> List[Path]:
        """
        Prepare files from directory for upload

        Args:
            source_dir: Directory containing files
            file_patterns: List of file patterns to include

        Returns:
            List of prepared file paths
        """
        try:
            source_dir = Path(source_dir)
            if not source_dir.exists():
                raise NotADirectoryError(f"Directory not found: {source_dir}")

            # Get default patterns if none provided
            if file_patterns is None:
                file_patterns = self.file_config['file_settings']['allowed_extensions']

            # Collect files matching patterns
            files_to_upload = []
            for pattern in file_patterns:
                files_to_upload.extend(source_dir.glob(f"*{pattern}"))

            # Prepare upload directory
            upload_dir = Path(self.file_config['file_settings']['uploads']['default_path'])
            upload_dir.mkdir(parents=True, exist_ok=True)

            # Process each file
            prepared_files = []
            for file_path in files_to_upload:
                try:
                    # Create copy in upload directory
                    target_path = upload_dir / file_path.name
                    shutil.copy2(file_path, target_path)

                    # Compress if needed
                    if self._should_compress(target_path):
                        target_path = self._compress_file(target_path)

                    prepared_files.append(target_path)
                    logger.info(f"Prepared {file_path.name} for upload")

                except Exception as e:
                    logger.error(f"Error preparing {file_path.name}: {str(e)}")

            return prepared_files

        except Exception as e:
            logger.error(f"Error preparing files: {str(e)}")
            raise

    def create_upload_manifest(
        self,
        prepared_files: List[Path]
    ) -> Dict:
        """
        Create manifest for uploaded files

        Args:
            prepared_files: List of prepared file paths

        Returns:
            Manifest dictionary
        """
        try:
            manifest = {
                'timestamp': datetime.now().isoformat(),
                'files': []
            }

            for file_path in prepared_files:
                file_info = {
                    'name': file_path.name,
                    'size': file_path.stat().st_size,
                    'compressed': file_path.suffix in ['.gz', '.zip'],
                    'path': str(file_path)
                }
                manifest['files'].append(file_info)

            # Save manifest
            manifest_dir = Path(self.file_config['file_settings']['uploads']['manifest_path'])
            manifest_dir.mkdir(parents=True, exist_ok=True)

            manifest_path = manifest_dir / f"upload_manifest_{datetime.now():%Y%m%d_%H%M%S}.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)

            return manifest

        except Exception as e:
            logger.error(f"Error creating manifest: {str(e)}")
            raise

def main():
    """Example usage of file upload operations"""
    try:
        # Initialize manager
        manager = FileUploadManager()

        # Prepare files for upload
        source_dir = Path("data/processed")
        prepared_files = manager.prepare_files_for_upload(
            source_dir,
            file_patterns=['.csv', '.json']
        )
        print(f"\nPrepared files: {prepared_files}")

        # Create manifest
        manifest = manager.create_upload_manifest(prepared_files)
        print("\nUpload manifest created")

        # Upload files
        results = manager.batch_upload_files(
            prepared_files,
            dataset_owner="username",
            dataset_name="example-dataset",
            target_directory="data"
        )

        print("\nUpload Results:")
        for result in results:
            status = result['status']
            file_name = result['file']
            print(f"{file_name}: {status}")

    except Exception as e:
        print(f"Error in file upload operations: {str(e)}")

if __name__ == "__main__":
    main()
