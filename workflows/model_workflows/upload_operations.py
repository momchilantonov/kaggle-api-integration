from pathlib import Path
from typing import Dict, Optional, Union, List, Callable
import logging
from datetime import datetime
import shutil
import json

from src.api.models import ModelClient, ModelMetadata
from src.utils.path_manager import PathManager
from src.utils.error_handlers import handle_api_errors
from src.handlers.data_handlers import DataHandler

logger = logging.getLogger(__name__)

class ModelUploadManager:
    """Manages model upload workflows"""

    def __init__(self):
        self.model_client = ModelClient()
        self.data_handler = DataHandler()
        self.path_manager = PathManager()
        # Ensure required directories exist
        self.path_manager.ensure_directories()

    @handle_api_errors
    def prepare_model_upload(
        self,
        model_path: Union[str, Path],
        framework: str,
        include_artifacts: bool = True,
        compression: bool = True
    ) -> Path:
        """Prepare model files for upload"""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model path not found: {model_path}")

            # Create upload directory
            upload_dir = (
                self.path_manager.get_path('models', 'uploads') /
                f"{model_path.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            upload_dir.mkdir(parents=True, exist_ok=True)

            # Copy model files based on framework
            self._copy_framework_files(model_path, upload_dir, framework)

            # Include training artifacts if requested
            if include_artifacts:
                self._copy_training_artifacts(model_path, upload_dir)

            # Create model card
            self._create_model_card(upload_dir, framework)

            # Compress if requested
            if compression:
                self._compress_large_files(upload_dir)

            logger.info(f"Prepared model upload at {upload_dir}")
            return upload_dir

        except Exception as e:
            logger.error(f"Error preparing model upload: {str(e)}")
            if 'upload_dir' in locals():
                shutil.rmtree(upload_dir)
            raise

    def _copy_framework_files(
        self,
        source_dir: Path,
        target_dir: Path,
        framework: str
    ) -> None:
        """Copy framework-specific model files"""
        try:
            # Define framework-specific file patterns
            framework_patterns = {
                'pytorch': ['.pt', '.pth'],
                'tensorflow': ['.pb', '.h5', '.keras'],
                'sklearn': ['.pkl', '.joblib']
            }

            # Copy model files
            patterns = framework_patterns.get(framework.lower(), ['.pkl'])
            files_copied = False
            for ext in patterns:
                for file_path in source_dir.glob(f'*{ext}'):
                    shutil.copy2(file_path, target_dir)
                    files_copied = True
                    logger.info(f"Copied {file_path.name}")

            # Copy configuration files
            for config_file in source_dir.glob('*.json'):
                shutil.copy2(config_file, target_dir)
                files_copied = True

            if not files_copied:
                raise ValueError(f"No valid model files found for framework {framework}")

        except Exception as e:
            logger.error(f"Error copying framework files: {str(e)}")
            raise

    def _copy_training_artifacts(
        self,
        source_dir: Path,
        target_dir: Path
    ) -> None:
        """Copy training artifacts"""
        try:
            artifacts_dir = target_dir / 'artifacts'
            artifacts_dir.mkdir(exist_ok=True)

            # Copy training logs
            log_dir = source_dir / 'logs'
            if log_dir.exists():
                shutil.copytree(log_dir, artifacts_dir / 'logs', dirs_exist_ok=True)

            # Copy metrics
            metrics_file = source_dir / 'metrics.json'
            if metrics_file.exists():
                shutil.copy2(metrics_file, artifacts_dir)

            # Copy visualizations
            vis_dir = source_dir / 'visualizations'
            if vis_dir.exists():
                shutil.copytree(vis_dir, artifacts_dir / 'visualizations', dirs_exist_ok=True)

        except Exception as e:
            logger.error(f"Error copying training artifacts: {str(e)}")
            raise

    def _create_model_card(
        self,
        model_dir: Path,
        framework: str
    ) -> None:
        """Create model card markdown file"""
        try:
            # Load metrics if available
            metrics = {}
            metrics_file = model_dir / 'artifacts' / 'metrics.json'
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)

            card_content = f"""
# Model Card

## Model Details
- Framework: {framework}
- Version: {datetime.now().strftime('%Y.%m.%d')}
- Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Architecture
[Model architecture details should be added here]

## Training Data
[Training data description should be added here]

## Performance Metrics
{self._format_metrics(metrics)}

## Limitations and Biases
[Model limitations and potential biases should be documented here]

## Additional Information
- Framework Version: [Framework version]
- Training Hardware: [Hardware details]
- Training Time: [Training duration]
            """

            card_path = model_dir / 'MODEL_CARD.md'
            with open(card_path, 'w') as f:
                f.write(card_content.strip())

        except Exception as e:
            logger.error(f"Error creating model card: {str(e)}")
            raise

    def _format_metrics(self, metrics: Dict) -> str:
        """Format metrics for model card"""
        if not metrics:
            return "No metrics available"

        formatted = "### Metrics\n"
        for metric, value in metrics.items():
            if isinstance(value, float):
                formatted += f"- {metric}: {value:.4f}\n"
            else:
                formatted += f"- {metric}: {value}\n"
        return formatted

    def _compress_large_files(self, directory: Path, threshold_mb: int = 100) -> None:
        """Compress large files in directory"""
        try:
            import gzip
            for file_path in directory.glob('**/*'):
                if file_path.is_file() and file_path.stat().st_size > threshold_mb * 1024 * 1024:
                    with open(file_path, 'rb') as f_in:
                        with gzip.open(f"{file_path}.gz", 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    file_path.unlink()  # Remove original file
                    logger.info(f"Compressed {file_path.name}")

        except Exception as e:
            logger.error(f"Error compressing files: {str(e)}")
            raise

    @handle_api_errors
    def upload_model(
        self,
        model_dir: Path,
        metadata: ModelMetadata,
        version_notes: Optional[str] = None,
        callback: Optional[Callable] = None
    ) -> Dict:
        """Upload model to Kaggle"""
        try:
            # Validate model directory
            if not model_dir.exists():
                raise NotADirectoryError(f"Model directory not found: {model_dir}")

            # Create backup
            backup_path = self._create_backup(model_dir)

            # Initialize model
            if callback:
                callback(20, "Initializing model")

            result = self.model_client.push_model(
                model_dir,
                metadata,
                version_notes=version_notes
            )

            if callback:
                callback(100, "Upload complete")

            # Log upload
            self._log_upload(metadata, result)

            return result

        except Exception as e:
            logger.error(f"Error uploading model: {str(e)}")
            raise

    def _create_backup(self, model_dir: Path) -> Path:
        """Create backup of model directory"""
        try:
            backup_dir = (
                self.path_manager.get_path('models', 'backups') /
                f"{model_dir.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            shutil.copytree(model_dir, backup_dir)
            logger.info(f"Created backup at {backup_dir}")
            return backup_dir

        except Exception as e:
            logger.error(f"Error creating backup: {str(e)}")
            raise

    def _log_upload(self, metadata: ModelMetadata, result: Dict) -> None:
        """Log model upload details"""
        try:
            log_path = self.path_manager.get_path('models', 'uploads') / 'uploads.log'
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'model_name': metadata.name,
                'version': metadata.version_name,
                'framework': metadata.framework,
                'result': result
            }

            with open(log_path, 'a') as f:
                f.write(f"{json.dumps(log_entry)}\n")

        except Exception as e:
            logger.error(f"Error logging upload: {str(e)}")

if __name__ == '__main__':
    # Example usage
    manager = ModelUploadManager()

    try:
        # Prepare model for upload
        model_path = Path("path/to/model")
        upload_dir = manager.prepare_model_upload(
            model_path,
            framework="pytorch",
            include_artifacts=True
        )
        print(f"Prepared model at: {upload_dir}")

        # Create metadata
        metadata = ModelMetadata(
            name="example-model",
            version_name="v1.0",
            description="Example PyTorch model",
            framework="pytorch",
            task_ids=["image-classification"]
        )

        # Upload model
        def progress_callback(progress, status):
            print(f"Progress: {progress}%, Status: {status}")

        result = manager.upload_model(
            upload_dir,
            metadata,
            callback=progress_callback
        )
        print(f"Upload result: {result}")

    except Exception as e:
        print(f"Error: {str(e)}")
