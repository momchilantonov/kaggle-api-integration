from pathlib import Path
from typing import Dict, Optional, Union, List
import logging
from datetime import datetime
import hashlib
import json

from src.api.files import FileClient
from src.utils.path_manager import PathManager
from src.utils.error_handlers import handle_api_errors
from src.handlers.data_handlers import DataHandler

logger = logging.getLogger(__name__)

class FileDownloadManager:
    """Manages file download workflows"""

    def __init__(self):
        self.file_client = FileClient()
        self.data_handler = DataHandler()
        self.path_manager = PathManager()
        # Ensure required directories exist
        self.path_manager.ensure_directories()

    @handle_api_errors
    def get_file(
        self,
        dataset_owner: str,
        dataset_name: str,
        file_name: str,
        custom_path: Optional[Path] = None,
        force: bool = False,
        verify: bool = True
    ) -> Path:
        """Download a specific file from a dataset"""
        try:
            # Determine download path
            file_path = (
                custom_path or
                self.path_manager.get_path('files', 'downloads') / file_name
            )

            # Check if file already exists
            if file_path.exists() and not force:
                logger.info(f"File already exists: {file_path}")
                return file_path

            # Create parent directories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Download file
            downloaded_path = self.file_client.get_file(
                dataset_owner=dataset_owner,
                dataset_name=dataset_name,
                file_name=file_name,
                path=file_path,
                force=force
            )

            # Verify download if requested
            if verify:
                self._verify_download(downloaded_path)

            # Log download
            self._log_download(dataset_owner, dataset_name, file_name, downloaded_path)

            return downloaded_path

        except Exception as e:
            logger.error(f"Error downloading file: {str(e)}")
            raise

    def _verify_download(self, file_path: Path) -> None:
        """Verify downloaded file integrity"""
        try:
            # Check file exists and is not empty
            if not file_path.exists():
                raise FileNotFoundError(f"Downloaded file not found: {file_path}")

            if file_path.stat().st_size == 0:
                raise ValueError(f"Downloaded file is empty: {file_path}")

            # Calculate file hash
            sha256_hash = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for block in iter(lambda: f.read(4096), b''):
                    sha256_hash.update(block)

            # Store hash in metadata
            metadata_path = file_path.parent / f"{file_path.name}.meta"
            metadata = {
                'filename': file_path.name,
                'size': file_path.stat().st_size,
                'hash': sha256_hash.hexdigest(),
                'download_date': datetime.now().isoformat()
            }

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            logger.error(f"Error verifying download: {str(e)}")
            raise

    def _log_download(
        self,
        dataset_owner: str,
        dataset_name: str,
        file_name: str,
        file_path: Path
    ) -> None:
        """Log file download details"""
        try:
            log_path = self.path_manager.get_path('files', 'downloads') / 'downloads.log'
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'dataset_owner': dataset_owner,
                'dataset_name': dataset_name,
                'file_name': file_name,
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size
            }

            with open(log_path, 'a') as f:
                f.write(f"{json.dumps(log_entry)}\n")

        except Exception as e:
            logger.error(f"Error logging download: {str(e)}")
            # Don't raise as this is non-critical

    @handle_api_errors
    def batch_download_files(
        self,
        file_list: List[Dict],
        output_dir: Optional[Path] = None,
        verify: bool = True
    ) -> Dict[str, Path]:
        """Download multiple files in batch"""
        try:
            results = {}
            failures = {}

            for file_info in file_list:
                try:
                    downloaded_path = self.get_file(
                        dataset_owner=file_info['owner'],
                        dataset_name=file_info['dataset'],
                        file_name=file_info['name'],
                        custom_path=output_dir / file_info['name'] if output_dir else None,
                        verify=verify
                    )
                    results[file_info['name']] = downloaded_path
                except Exception as e:
                    failures[file_info['name']] = str(e)
                    logger.error(f"Error downloading {file_info['name']}: {str(e)}")

            if failures:
                logger.warning(f"Some files failed to download: {failures}")

            return {
                'successful': results,
                'failed': failures
            }

        except Exception as e:
            logger.error(f"Error in batch download: {str(e)}")
            raise

    @handle_api_errors
    def verify_downloads(
        self,
        directory: Optional[Path] = None,
        fix_issues: bool = False
    ) -> Dict:
        """Verify integrity of downloaded files"""
        try:
            directory = directory or self.path_manager.get_path('files', 'downloads')
            verification_results = {
                'verified': [],
                'failed': [],
                'fixed': []
            }

            for file_path in directory.glob('**/*'):
                if file_path.suffix == '.meta':
                    continue

                meta_path = file_path.parent / f"{file_path.name}.meta"
                if not meta_path.exists():
                    verification_results['failed'].append({
                        'file': str(file_path),
                        'reason': 'Missing metadata'
                    })
                    continue

                try:
                    # Verify against metadata
                    with open(meta_path) as f:
                        metadata = json.load(f)

                    # Calculate current hash
                    sha256_hash = hashlib.sha256()
                    with open(file_path, 'rb') as f:
                        for block in iter(lambda: f.read(4096), b''):
                            sha256_hash.update(block)
                    current_hash = sha256_hash.hexdigest()

                    if current_hash != metadata['hash']:
                        if fix_issues:
                            # Re-download file
                            self.get_file(
                                dataset_owner=metadata.get('dataset_owner'),
                                dataset_name=metadata.get('dataset_name'),
                                file_name=file_path.name,
                                custom_path=file_path,
                                force=True
                            )
                            verification_results['fixed'].append(str(file_path))
                        else:
                            verification_results['failed'].append({
                                'file': str(file_path),
                                'reason': 'Hash mismatch'
                            })
                    else:
                        verification_results['verified'].append(str(file_path))

                except Exception as e:
                    verification_results['failed'].append({
                        'file': str(file_path),
                        'reason': str(e)
                    })

            return verification_results

        except Exception as e:
            logger.error(f"Error verifying downloads: {str(e)}")
            raise

if __name__ == '__main__':
    # Example usage
    manager = FileDownloadManager()

    try:
        # Download single file
        file_path = manager.get_file(
            "owner",
            "dataset-name",
            "example.csv"
        )
        print(f"Downloaded file to: {file_path}")

        # Batch download
        files_to_download = [
            {'owner': 'owner1', 'dataset': 'dataset1', 'name': 'file1.csv'},
            {'owner': 'owner2', 'dataset': 'dataset2', 'name': 'file2.csv'}
        ]
        results = manager.batch_download_files(files_to_download)
        print(f"Batch download results: {results}")

        # Verify downloads
        verification = manager.verify_downloads(fix_issues=True)
        print(f"Verification results: {verification}")

    except Exception as e:
        print(f"Error: {str(e)}")
