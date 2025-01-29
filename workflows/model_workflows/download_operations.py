import os
from pathlib import Path
from typing import Dict, Optional, Union, List
import logging
from datetime import datetime
import json
import shutil

from src.api.models import ModelClient, ModelMetadata
from src.utils.path_manager import PathManager
from src.utils.error_handlers import handle_api_errors
from src.handlers.data_handlers import DataHandler

logger = logging.getLogger(__name__)

class ModelDownloadManager:
    """Manages model download workflows"""

    def __init__(self):
        self.model_client = ModelClient()
        self.data_handler = DataHandler()
        self.path_manager = PathManager()
        # Ensure required directories exist
        self.path_manager.ensure_directories()

    @handle_api_errors
    def download_model(
        self,
        owner: str,
        model_name: str,
        version: Optional[str] = None,
        custom_path: Optional[Path] = None,
        extract: bool = True
    ) -> Path:
        """Download model files"""
        try:
            # Get model path
            model_path = (
                custom_path or
                self.path_manager.get_path('models', 'downloaded') / f"{owner}_{model_name}"
            )
            model_path.mkdir(parents=True, exist_ok=True)

            # Download model
            downloaded_path = self.model_client.pull_model(
                owner=owner,
                model_name=model_name,
                version=version,
                path=model_path
            )

            # Extract if requested and file is compressed
            if extract and downloaded_path.suffix == '.zip':
                self._extract_model(downloaded_path, model_path)
                downloaded_path.unlink()  # Remove zip after extraction

            # Create download metadata
            self._create_download_metadata(
                model_path,
                owner,
                model_name,
                version
            )

            logger.info(f"Downloaded model to {model_path}")
            return model_path

        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            raise

    def _extract_model(self, zip_path: Path, extract_path: Path) -> None:
        """Extract downloaded model files"""
        try:
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            logger.info(f"Extracted model files to {extract_path}")

        except Exception as e:
            logger.error(f"Error extracting model files: {str(e)}")
            raise

    def _create_download_metadata(
        self,
        model_path: Path,
        owner: str,
        model_name: str,
        version: Optional[str]
    ) -> None:
        """Create metadata about downloaded model"""
        try:
            metadata = {
                'owner': owner,
                'model_name': model_name,
                'version': version,
                'download_date': datetime.now().isoformat(),
                'files': [
                    {
                        'name': f.name,
                        'size': f.stat().st_size,
                        'type': f.suffix[1:] if f.suffix else 'unknown'
                    }
                    for f in model_path.glob('**/*')
                    if f.is_file()
                ]
            }

            # Get model info if available
            try:
                model_info = self.get_model_info(model_path)
                metadata['model_info'] = model_info
            except Exception:
                pass

            metadata_path = model_path / 'download_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            logger.error(f"Error creating download metadata: {str(e)}")
            raise

    @handle_api_errors
    def get_model_info(self, model_path: Path) -> Dict:
        """Extract model information from downloaded files"""
        try:
            info = {}

            # Check for model card
            model_card = model_path / 'MODEL_CARD.md'
            if model_card.exists():
                info['model_card'] = model_card.read_text()

            # Check for metrics
            metrics_file = model_path / 'metrics.json'
            if metrics_file.exists():
                with open(metrics_file) as f:
                    info['metrics'] = json.load(f)

            # Check for config
            config_file = model_path / 'config.json'
            if config_file.exists():
                with open(config_file) as f:
                    info['config'] = json.load(f)

            return info

        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            raise

    @handle_api_errors
    def list_model_versions(
        self,
        owner: str,
        model_name: str
    ) -> List[Dict]:
        """List all versions of a model"""
        try:
            versions = self.model_client.list_model_versions(owner, model_name)

            # Enhance version information
            enhanced_versions = []
            for version in versions:
                enhanced_version = {
                    'version_number': version['version_number'],
                    'created': version['created'],
                    'framework': version.get('framework', 'unknown'),
                    'size': self._format_size(version.get('size', 0)),
                    'description': version.get('description', ''),
                    'is_latest': version.get('is_latest', False)
                }
                enhanced_versions.append(enhanced_version)

            return enhanced_versions

        except Exception as e:
            logger.error(f"Error listing model versions: {str(e)}")
            raise

    def _format_size(self, size_bytes: int) -> str:
        """Format size in bytes to human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} TB"

    @handle_api_errors
    def verify_downloaded_model(
        self,
        model_path: Path,
        framework: str
    ) -> Dict:
        """Verify downloaded model files and structure"""
        try:
            verification = {
                'status': 'success',
                'checks': [],
                'warnings': []
            }

            # Check framework-specific files
            framework_files = {
                'pytorch': ['.pt', '.pth'],
                'tensorflow': ['.pb', '.h5', '.keras'],
                'sklearn': ['.pkl', '.joblib']
            }

            required_extensions = framework_files.get(framework.lower(), [])
            found_model_file = False
            for ext in required_extensions:
                if list(model_path.glob(f'**/*{ext}')):
                    found_model_file = True
                    verification['checks'].append(f"Found model file with extension {ext}")
                    break

            if not found_model_file:
                verification['status'] = 'warning'
                verification['warnings'].append(
                    f"No model file found with extensions {required_extensions}"
                )

            # Check for required metadata files
            required_files = ['MODEL_CARD.md', 'download_metadata.json']
            for file in required_files:
                if (model_path / file).exists():
                    verification['checks'].append(f"Found {file}")
                else:
                    verification['warnings'].append(f"Missing {file}")

            # Check file permissions
            for file_path in model_path.glob('**/*'):
                if not os.access(file_path, os.R_OK):
                    verification['warnings'].append(
                        f"File {file_path.name} has incorrect permissions"
                    )

            return verification

        except Exception as e:
            logger.error(f"Error verifying model: {str(e)}")
            raise

    @handle_api_errors
    def clean_old_downloads(
        self,
        keep_days: int = 30,
        dry_run: bool = True
    ) -> List[Dict]:
        """Clean old downloaded models"""
        try:
            downloads_dir = self.path_manager.get_path('models', 'downloaded')
            cleanup_list = []
            current_time = datetime.now()

            for model_dir in downloads_dir.iterdir():
                if not model_dir.is_dir():
                    continue

                metadata_file = model_dir / 'download_metadata.json'
                if not metadata_file.exists():
                    continue

                with open(metadata_file) as f:
                    metadata = json.load(f)

                download_date = datetime.fromisoformat(metadata['download_date'])
                age_days = (current_time - download_date).days

                if age_days > keep_days:
                    cleanup_info = {
                        'path': str(model_dir),
                        'age_days': age_days,
                        'size': sum(f.stat().st_size for f in model_dir.glob('**/*') if f.is_file())
                    }
                    cleanup_list.append(cleanup_info)

                    if not dry_run:
                        shutil.rmtree(model_dir)
                        logger.info(f"Removed old model directory: {model_dir}")

            return cleanup_list

        except Exception as e:
            logger.error(f"Error cleaning old downloads: {str(e)}")
            raise

if __name__ == '__main__':
    # Example usage
    manager = ModelDownloadManager()

    try:
        # Download model
        model_path = manager.download_model(
            "owner",
            "model-name",
            version="latest"
        )
        print(f"Downloaded model to: {model_path}")

        # Verify download
        verification = manager.verify_downloaded_model(
            model_path,
            framework="pytorch"
        )
        print(f"Verification result: {verification}")

        # List versions
        versions = manager.list_model_versions("owner", "model-name")
        print("Model versions:")
        for version in versions:
            print(f"- Version {version['version_number']}: {version['created']}")

        # Clean old downloads
        cleanup_list = manager.clean_old_downloads(keep_days=30, dry_run=True)
        print("\nPotential cleanup candidates:")
        for item in cleanup_list:
            print(f"- {item['path']} (Age: {item['age_days']} days)")

    except Exception as e:
        print(f"Error: {str(e)}")
