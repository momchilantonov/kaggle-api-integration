from pathlib import Path
from typing import Dict, Optional, Union, List, Callable
import logging
from datetime import datetime
import hashlib
import json
import shutil

from src.api.files import FileClient
from src.utils.path_manager import PathManager
from src.utils.error_handlers import handle_api_errors
from src.handlers.data_handlers import DataHandler

logger = logging.getLogger(__name__)

class FileUploadManager:
    """Manages file upload workflows"""

    def __init__(self):
        self.file_client = FileClient()
        self.data_handler = DataHandler()
        self.path_manager = PathManager()
        # Ensure required directories exist
        self.path_manager.ensure_directories()

    @handle_api_errors
    def upload_file(
        self,
        file_path: Union[str, Path],
        dataset_owner: str,
        dataset_name: str,
        target_path: Optional[str] = None,
        validate: bool = True,
        backup: bool = True,
        callback: Optional[Callable] = None
    ) -> Dict:
        """Upload a file to a dataset"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Validate file if requested
            if validate:
                self._validate_file(file_path)

            # Create backup if requested
            if backup:
                backup_path = self._create_backup(file_path)
                logger.info(f"Created backup at: {backup_path}")

            # Progress update
            if callback:
                callback(20, "File validated and backed up")

            # Calculate file hash before upload
            file_hash = self._calculate_file_hash(file_path)

            # Upload file
            result = self.file_client.upload_file(
                dataset_owner=dataset_owner,
                dataset_name=dataset_name,
                file_path=file_path,
                target_path=target_path
            )

            # Progress update
            if callback:
                callback(80, "File uploaded")

            # Log upload
            self._log_upload(dataset_owner, dataset_name, file_path, file_hash, result)

            # Progress update
            if callback:
                callback(100, "Upload completed")

            return result

        except Exception as e:
            logger.error(f"Error uploading file: {str(e)}")
            raise

    def _validate_file(self, file_path: Path) -> None:
        """Validate file before upload"""
        try:
            # Check file size (e.g., 100MB limit)
            max_size = 100 * 1024 * 1024  # 100MB
            if file_path.stat().st_size > max_size:
                raise ValueError(f"File {file_path.name} exceeds size limit of 100MB")

            # Validate file type
            allowed_extensions = {'.csv', '.json', '.txt', '.zip', '.gz', '.parquet'}
            if file_path.suffix.lower() not in allowed_extensions:
                raise ValueError(
                    f"File type {file_path.suffix} not allowed. "
                    f"Allowed types: {allowed_extensions}"
                )

            # Validate CSV files
            if file_path.suffix.lower() == '.csv':
                df = self.data_handler.read_csv(file_path)
                if df.empty:
                    raise ValueError(f"CSV file {file_path.name} is empty")

        except Exception as e:
            logger.error(f"Error validating file: {str(e)}")
            raise

    def _create_backup(self, file_path: Path) -> Path:
        """Create backup of file before upload"""
        try:
            backup_dir = self.path_manager.get_path('files', 'backups')
            backup_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"{file_path.stem}_{timestamp}{file_path.suffix}"

            shutil.copy2(file_path, backup_path)
            return backup_path

        except Exception as e:
            logger.error(f"Error creating backup: {str(e)}")
            raise

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for block in iter(lambda: f.read(4096), b''):
                    sha256_hash.update(block)
            return sha256_hash.hexdigest()

        except Exception as e:
            logger.error(f"Error calculating file hash: {str(e)}")
            raise

    def _log_upload(
        self,
        dataset_owner: str,
        dataset_name: str,
        file_path: Path,
        file_hash: str,
        result: Dict
    ) -> None:
        """Log file upload details"""
        try:
            log_path = self.path_manager.get_path('files', 'uploads') / 'uploads.log'
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'dataset_owner': dataset_owner,
                'dataset_name': dataset_name,
                'file_name': file_path.name,
                'file_size': file_path.stat().st_size,
                'file_hash': file_hash,
                'upload_result': result
            }

            with open(log_path, 'a') as f:
                f.write(f"{json.dumps(log_entry)}\n")

        except Exception as e:
            logger.error(f"Error logging upload: {str(e)}")

    @handle_api_errors
    def batch_upload_files(
        self,
        file_list: List[Dict],
        validate: bool = True,
        backup: bool = True
    ) -> Dict[str, Dict]:
        """Upload multiple files in batch"""
        try:
            results = {}
            failures = {}

            for file_info in file_list:
                try:
                    result = self.upload_file(
                        file_path=file_info['path'],
                        dataset_owner=file_info['owner'],
                        dataset_name=file_info['dataset'],
                        target_path=file_info.get('target_path'),
                        validate=validate,
                        backup=backup
                    )
                    results[file_info['path']] = result
                except Exception as e:
                    failures[file_info['path']] = str(e)
                    logger.error(f"Error uploading {file_info['path']}: {str(e)}")

            return {
                'successful': results,
                'failed': failures
            }

        except Exception as e:
            logger.error(f"Error in batch upload: {str(e)}")
            raise

    @handle_api_errors
    def delete_file(
        self,
        dataset_owner: str,
        dataset_name: str,
        file_name: str,
        backup: bool = True
    ) -> Dict:
        """Delete a file from a dataset"""
        try:
            # Get file details before deletion
            file_info = self.file_client.get_file_info(
                dataset_owner,
                dataset_name,
                file_name
            )

            # Create backup if requested
            if backup and file_info:
                self._backup_before_deletion(
                    dataset_owner,
                    dataset_name,
                    file_name,
                    file_info
                )

            # Delete file
            result = self.file_client.delete_file(
                dataset_owner=dataset_owner,
                dataset_name=dataset_name,
                file_name=file_name
            )

            # Log deletion
            self._log_deletion(dataset_owner, dataset_name, file_name, result)

            return result

        except Exception as e:
            logger.error(f"Error deleting file: {str(e)}")
            raise

    def _backup_before_deletion(
        self,
        dataset_owner: str,
        dataset_name: str,
        file_name: str,
        file_info: Dict
    ) -> None:
        """Create backup before file deletion"""
        try:
            backup_dir = self.path_manager.get_path('files', 'backups') / 'deleted'
            backup_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_info = {
                'timestamp': timestamp,
                'dataset_owner': dataset_owner,
                'dataset_name': dataset_name,
                'file_name': file_name,
                'file_info': file_info
            }

            backup_meta_path = backup_dir / f"{file_name}_{timestamp}.meta"
            with open(backup_meta_path, 'w') as f:
                json.dump(backup_info, f, indent=2)

        except Exception as e:
            logger.error(f"Error creating deletion backup: {str(e)}")
            # Continue with deletion even if backup fails

    def _log_deletion(
        self,
        dataset_owner: str,
        dataset_name: str,
        file_name: str,
        result: Dict
    ) -> None:
        """Log file deletion details"""
        try:
            log_path = self.path_manager.get_path('files', 'uploads') / 'deletions.log'
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'dataset_owner': dataset_owner,
                'dataset_name': dataset_name,
                'file_name': file_name,
                'deletion_result': result
            }

            with open(log_path, 'a') as f:
                f.write(f"{json.dumps(log_entry)}\n")

        except Exception as e:
            logger.error(f"Error logging deletion: {str(e)}")

if __name__ == '__main__':
    # Example usage
    manager = FileUploadManager()

    try:
        # Upload single file
        result = manager.upload_file(
            "path/to/file.csv",
            "dataset-owner",
            "dataset-name",
            validate=True,
            backup=True
        )
        print(f"Upload result: {result}")

        # Batch upload
        files_to_upload = [
            {
                'path': 'path/to/file1.csv',
                'owner': 'owner1',
                'dataset': 'dataset1'
            },
            {
                'path': 'path/to/file2.csv',
                'owner': 'owner2',
                'dataset': 'dataset2'
            }
        ]
        results = manager.batch_upload_files(files_to_upload)
        print(f"Batch upload results: {results}")

        # Delete file
        delete_result = manager.delete_file(
            "dataset-owner",
            "dataset-name",
            "file.csv",
            backup=True
        )
        print(f"Delete result: {delete_result}")

    except Exception as e:
        print(f"Error: {str(e)}")
