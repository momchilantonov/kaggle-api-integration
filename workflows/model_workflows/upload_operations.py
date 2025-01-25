from pathlib import Path
import yaml
from typing import Optional, List, Dict, Union
import logging
import shutil
import json
from datetime import datetime

from src.api.kaggle_client import KaggleAPIClient
from src.api.models import ModelClient, ModelMetadata
from src.utils.helpers import timer, retry_on_exception, compress_file

logger = logging.getLogger(__name__)

class ModelUploadManager:
    def __init__(self):
        """Initialize the model upload manager"""
        self.kaggle_client = KaggleAPIClient()
        self.model_client = ModelClient(self.kaggle_client)
        self._load_configs()

    def _load_configs(self):
        """Load operational configurations"""
        try:
            with open('operational_configs/model_configs/model_params.yaml', 'r') as f:
                self.model_config = yaml.safe_load(f)
            with open('operational_configs/model_configs/training_config.yaml', 'r') as f:
                self.training_config = yaml.safe_load(f)
            logger.info("Successfully loaded model configurations")
        except Exception as e:
            logger.error(f"Error loading configurations: {str(e)}")
            raise

    @timer
    def prepare_model_upload(
        self,
        model_path: Union[str, Path],
        framework: str,
        include_artifacts: bool = True
    ) -> Path:
        """
        Prepare model files for upload

        Args:
            model_path: Path to model directory
            framework: Model framework (PyTorch, TensorFlow, etc.)
            include_artifacts: Whether to include training artifacts

        Returns:
            Path to prepared upload directory
        """
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model path not found: {model_path}")

            # Create upload directory
            upload_dir = Path("data/models/uploads") / f"{model_path.name}_upload"
            upload_dir.mkdir(parents=True, exist_ok=True)

            # Copy model files based on framework
            self._copy_framework_files(model_path, upload_dir, framework)

            # Include additional artifacts if requested
            if include_artifacts:
                self._copy_training_artifacts(model_path, upload_dir)

            # Create model card
            self._create_model_card(upload_dir, framework)

            logger.info(f"Prepared model upload at {upload_dir}")
            return upload_dir

        except Exception as e:
            logger.error(f"Error preparing model upload: {str(e)}")
            raise

    def _copy_framework_files(
        self,
        source_dir: Path,
        target_dir: Path,
        framework: str
    ) -> None:
        """Copy framework-specific model files"""
        try:
            framework = framework.lower()
            if framework == 'pytorch':
                extensions = ['.pt', '.pth']
            elif framework == 'tensorflow':
                extensions = ['.pb', '.h5', '.keras']
            else:
                extensions = ['.pkl', '.joblib']

            # Copy model files
            for ext in extensions:
                for file_path in source_dir.glob(f"*{ext}"):
                    shutil.copy2(file_path, target_dir)
                    logger.info(f"Copied {file_path.name}")

            # Copy configuration files
            for config_file in source_dir.glob('*.json'):
                shutil.copy2(config_file, target_dir)

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
    ) -> Path:
        """Create model card markdown file"""
        try:
            card_content = f"""
# Model Card

## Model Details
- Framework: {framework}
- Version: {datetime.now().strftime('%Y.%m.%d')}
- Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Architecture
[Describe the model architecture here]

## Training Data
[Describe the training data here]

## Performance Metrics
[Add performance metrics here]

## Limitations and Biases
[Describe any known limitations or biases]

## Additional Information
- Framework Version: [Add framework version]
- Training Hardware: [Add training hardware details]
- Training Time: [Add training time]
            """

            card_path = model_dir / 'MODEL_CARD.md'
            card_path.write_text(card_content.strip())
            return card_path

        except Exception as e:
            logger.error(f"Error creating model card: {str(e)}")
            raise

    @timer
    @retry_on_exception(retries=3, delay=1)
    def upload_model(
        self,
        model_dir: Path,
        metadata: ModelMetadata,
        wait_for_processing: bool = True
    ) -> Dict:
        """
        Upload model to Kaggle

        Args:
            model_dir: Directory containing model files
            metadata: Model metadata
            wait_for_processing: Whether to wait for processing

        Returns:
            Upload response
        """
        try:
            result = self.model_client.push_model(
                model_dir,
                metadata
            )

            if wait_for_processing:
                result = self.model_client.wait_for_model_ready(
                    metadata.name,
                    timeout=self.model_config.get('upload_timeout', 3600)
                )

            logger.info(f"Successfully uploaded model: {result}")
            return result

        except Exception as e:
            logger.error(f"Error uploading model: {str(e)}")
            raise

    @timer
    def create_model_version(
        self,
        model_dir: Path,
        version_notes: str,
        metadata: Optional[ModelMetadata] = None
    ) -> Dict:
        """
        Create new version of existing model

        Args:
            model_dir: Directory containing model files
            version_notes: Notes for version
            metadata: Optional updated metadata

        Returns:
            Version creation response
        """
        try:
            if metadata:
                result = self.model_client.push_model(
                    model_dir,
                    metadata,
                    version_notes=version_notes
                )
            else:
                # Use existing metadata with new version
                result = self.model_client.create_version(
                    model_dir,
                    version_notes=version_notes
                )

            logger.info(f"Created new model version: {result}")
            return result

        except Exception as e:
            logger.error(f"Error creating model version: {str(e)}")
            raise

def main():
    """Example usage of model upload operations"""
    try:
        # Initialize manager
        manager = ModelUploadManager()

        # Prepare model for upload
        model_path = Path("data/models/custom/example_model")
        if model_path.exists():
            upload_dir = manager.prepare_model_upload(
                model_path,
                framework="PyTorch",
                include_artifacts=True
            )
            print(f"\nPrepared model upload at: {upload_dir}")

            # Create metadata
            metadata = ModelMetadata(
                name="example-model",
                version_name="v1.0",
                description="Example PyTorch model",
                framework="PyTorch",
                task_ids=["image-classification"],
                training_data="Custom dataset"
            )

            # Upload model
            result = manager.upload_model(
                upload_dir,
                metadata,
                wait_for_processing=True
            )
            print(f"\nUpload result: {result}")

            # Create new version
            version_result = manager.create_model_version(
                upload_dir,
                "Updated model weights",
                metadata=metadata
            )
            print(f"\nVersion creation result: {version_result}")

    except Exception as e:
        print(f"Error in model upload operations: {str(e)}")

if __name__ == "__main__":
    main()
