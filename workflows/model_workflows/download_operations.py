from pathlib import Path
import yaml
from typing import Optional, List, Dict, Union
import logging
import time

from src.api.kaggle_client import KaggleAPIClient
from src.api.models import ModelClient, ModelMetadata
from src.utils.helpers import timer, retry_on_exception, memory_monitor

logger = logging.getLogger(__name__)

class ModelWorkflowManager:
    def __init__(self):
        """Initialize the model workflow manager"""
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
    @retry_on_exception(retries=3, delay=1)
    def download_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        custom_path: Optional[Path] = None
    ) -> Path:
        """
        Download a model using predefined configurations

        Args:
            model_name: Name of the model from config
            version: Specific version to download
            custom_path: Optional custom download path

        Returns:
            Path to downloaded model
        """
        try:
            # Get model info from config
            model_info = self.model_config['model_settings'].get(model_name)
            if not model_info:
                raise ValueError(f"Model {model_name} not found in configurations")

            # Determine download path
            base_path = custom_path or Path(model_info['local_path'])
            download_path = base_path / (version or 'latest')
            download_path.mkdir(parents=True, exist_ok=True)

            # Download model
            model_path = self.model_client.pull_model(
                owner=model_info['owner'],
                model_name=model_name,
                version=version,
                path=download_path
            )

            logger.info(f"Successfully downloaded model {model_name} to {model_path}")
            return model_path

        except Exception as e:
            logger.error(f"Error downloading model {model_name}: {str(e)}")
            raise

    @timer
    @memory_monitor(threshold_mb=1000)
    def verify_model_files(
        self,
        model_path: Path,
        expected_files: List[str]
    ) -> Dict[str, bool]:
        """
        Verify that all expected model files are present

        Args:
            model_path: Path to downloaded model
            expected_files: List of expected file names

        Returns:
            Dictionary indicating presence of each file
        """
        results = {}
        for file_name in expected_files:
            file_path = model_path / file_name
            results[file_name] = file_path.exists()
            if not file_path.exists():
                logger.warning(f"Missing expected file: {file_name}")
        return results

    @timer
    def push_model_version(
        self,
        model_path: Path,
        metadata: ModelMetadata,
        wait_for_completion: bool = True
    ) -> Dict:
        """
        Push a new model version

        Args:
            model_path: Path to model files
            metadata: Model metadata
            wait_for_completion: Whether to wait for processing completion

        Returns:
            Response from the API
        """
        try:
            # Push model
            result = self.model_client.push_model(
                model_path,
                metadata,
                public=self.model_config.get('default_visibility', True)
            )

            if wait_for_completion:
                result = self.wait_for_model_ready(
                    metadata.name,
                    timeout=self.model_config.get('push_timeout', 3600)
                )

            logger.info(f"Successfully pushed model version: {result}")
            return result

        except Exception as e:
            logger.error(f"Error pushing model version: {str(e)}")
            raise

    def wait_for_model_ready(
        self,
        model_name: str,
        timeout: int = 3600,
        check_interval: int = 10
    ) -> Dict:
        """
        Wait for model processing to complete

        Args:
            model_name: Name of the model
            timeout: Maximum time to wait in seconds
            check_interval: Time between status checks

        Returns:
            Final model status
        """
        start_time = time.time()
        while True:
            try:
                status = self.model_client.get_model_status(model_name)

                if status.get('status') == 'complete':
                    logger.info(f"Model {model_name} is ready")
                    return status

                if status.get('status') == 'failed':
                    error_msg = status.get('errorMessage', 'Unknown error')
                    raise RuntimeError(f"Model processing failed: {error_msg}")

                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Model not ready after {timeout} seconds")

                time.sleep(check_interval)

            except Exception as e:
                logger.error(f"Error checking model status: {str(e)}")
                raise

def main():
    """Example usage of model workflows"""
    try:
        # Initialize manager
        manager = ModelWorkflowManager()

        # Download specific model
        model_path = manager.download_model(
            model_name="resnet50",
            version="latest"
        )
        print(f"\nDownloaded model to: {model_path}")

        # Verify model files
        expected_files = [
            "model.pth",
            "config.json",
            "README.md"
        ]
        verification_results = manager.verify_model_files(
            model_path,
            expected_files
        )
        print("\nFile Verification Results:")
        for file_name, exists in verification_results.items():
            print(f"{file_name}: {'Present' if exists else 'Missing'}")

        # Example of pushing a new model version
        metadata = ModelMetadata(
            name="custom-resnet",
            version_name="v1.0",
            description="Custom ResNet model for image classification",
            framework="PyTorch",
            task_ids=["computer-vision", "image-classification"],
            training_data="ImageNet"
        )

        new_model_path = Path("data/models/custom/resnet-modified")
        if new_model_path.exists():
            result = manager.push_model_version(
                new_model_path,
                metadata,
                wait_for_completion=True
            )
            print(f"\nPush result: {result}")

    except Exception as e:
        print(f"Error in model workflow: {str(e)}")

if __name__ == "__main__":
    main()
