from pathlib import Path
from typing import Dict, Optional, Union, List, Callable
import logging
from datetime import datetime
import json
import shutil

from src.api.datasets import DatasetClient, DatasetMetadata
from src.handlers.data_handlers import DataHandler
from src.utils.path_manager import PathManager
from src.utils.error_handlers import handle_api_errors

logger = logging.getLogger(__name__)

class DatasetUploadManager:
    """Manages dataset upload workflows"""

    def __init__(self):
        self.dataset_client = DatasetClient()
        self.data_handler = DataHandler()
        self.path_manager = PathManager()
        # Ensure required directories exist
        self.path_manager.ensure_directories()

    @handle_api_errors
    def prepare_dataset_folder(
        self,
        source_files: List[Union[str, Path]],
        dataset_name: str,
        include_metadata: bool = True,
        validate_files: bool = True
    ) -> Path:
        """Prepare a folder with dataset files and metadata"""
        try:
            # Get upload directory path
            upload_dir = self.path_manager.get_path('datasets', 'uploads') / dataset_name
            upload_dir.mkdir(parents=True, exist_ok=True)

            # Validate and copy files
            copied_files = []
            total_size = 0
            for file_path in source_files:
                file_path = Path(file_path)
                if not file_path.exists():
                    raise FileNotFoundError(f"File not found: {file_path}")

                if validate_files:
                    self._validate_file(file_path)

                # Copy file
                target_path = upload_dir / file_path.name
                shutil.copy2(file_path, target_path)
                copied_files.append({
                    'name': file_path.name,
                    'size': file_path.stat().st_size,
                    'type': file_path.suffix[1:] if file_path.suffix else 'unknown'
                })
                total_size += file_path.stat().st_size

                logger.info(f"Copied {file_path.name} to dataset directory")

            # Create dataset metadata if requested
            if include_metadata:
                metadata = {
                    'name': dataset_name,
                    'created_at': datetime.now().isoformat(),
                    'files': copied_files,
                    'total_size': total_size,
                    'file_count': len(copied_files)
                }
                metadata_path = upload_dir / 'dataset-metadata.json'
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

            return upload_dir

        except Exception as e:
            logger.error(f"Error preparing dataset folder: {str(e)}")
            # Clean up on error
            if 'upload_dir' in locals():
                shutil.rmtree(upload_dir)
            raise

    def _validate_file(self, file_path: Path) -> None:
        """Validate file before upload"""
        # Check file size (100MB limit for example)
        max_size = 100 * 1024 * 1024  # 100MB
        if file_path.stat().st_size > max_size:
            raise ValueError(f"File {file_path.name} exceeds size limit of 100MB")

        # Validate file type
        allowed_extensions = {'.csv', '.json', '.txt', '.zip', '.gz'}
        if file_path.suffix.lower() not in allowed_extensions:
            raise ValueError(
                f"File type {file_path.suffix} not allowed. "
                f"Allowed types: {allowed_extensions}"
            )

        # Validate CSV files
        if file_path.suffix.lower() == '.csv':
            try:
                df = self.data_handler.read_csv(file_path)
                if df.empty:
                    raise ValueError(f"CSV file {file_path.name} is empty")
            except Exception as e:
                raise ValueError(f"Invalid CSV file {file_path.name}: {str(e)}")

    @handle_api_errors
    def upload_dataset(
        self,
        dataset_dir: Path,
        metadata: DatasetMetadata,
        public: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """Upload dataset to Kaggle"""
        try:
            # Validate dataset directory
            if not dataset_dir.exists():
                raise NotADirectoryError(f"Dataset directory not found: {dataset_dir}")

            # Create backup before upload
            backup_dir = self.path_manager.get_path('datasets', 'uploads') / 'backups'
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / f"{dataset_dir.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copytree(dataset_dir, backup_path)

            # Upload dataset
            result = self.dataset_client.create_dataset(
                folder_path=dataset_dir,
                metadata=metadata,
                public=public
            )

            # Log upload details
            upload_log_path = self.path_manager.get_path('datasets', 'uploads') / 'uploads.log'
            with open(upload_log_path, 'a') as f:
                f.write(
                    f"{datetime.now()},{metadata.title},{result.get('ref', 'N/A')},"
                    f"{'public' if public else 'private'}\n"
                )

            # Update progress if callback provided
            if progress_callback:
                progress_callback(100, 100)

            return result

        except Exception as e:
            logger.error(f"Error uploading dataset: {str(e)}")
            raise

    @handle_api_errors
    def create_dataset_version(
        self,
        dataset_dir: Path,
        version_notes: str,
        delete_old_versions: bool = False
    ) -> Dict:
        """Create new version of existing dataset"""
        try:
            # Validate dataset directory
            if not dataset_dir.exists():
                raise NotADirectoryError(f"Dataset directory not found: {dataset_dir}")

            # Create backup before version creation
            backup_dir = self.path_manager.get_path('datasets', 'uploads') / 'version_backups'
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / f"{dataset_dir.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copytree(dataset_dir, backup_path)

            # Create new version
            result = self.dataset_client.create_version(
                folder_path=dataset_dir,
                version_notes=version_notes,
                delete_old_versions=delete_old_versions
            )

            # Log version creation
            version_log_path = self.path_manager.get_path('datasets', 'uploads') / 'versions.log'
            with open(version_log_path, 'a') as f:
                f.write(
                    f"{datetime.now()},{dataset_dir.name},{result.get('version', 'N/A')},"
                    f"{version_notes}\n"
                )

            return result

        except Exception as e:
            logger.error(f"Error creating dataset version: {str(e)}")
            raise

if __name__ == '__main__':
    # Example usage
    manager = DatasetUploadManager()

    try:
        # Prepare dataset for upload
        source_files = [Path("data.csv"), Path("description.txt")]
        dataset_dir = manager.prepare_dataset_folder(source_files, "example-dataset")
        print(f"Prepared dataset at: {dataset_dir}")

        # Create metadata
        metadata = DatasetMetadata(
            title="Example Dataset",
            slug="example-dataset",
            description="This is an example dataset",
            licenses=[{"name": "CC0-1.0"}],
            keywords=["example", "test"]
        )

        # Upload dataset
        result = manager.upload_dataset(dataset_dir, metadata)
        print(f"Upload result: {result}")

    except Exception as e:
        print(f"Error: {str(e)}")
