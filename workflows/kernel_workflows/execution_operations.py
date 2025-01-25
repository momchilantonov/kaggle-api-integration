from pathlib import Path
import yaml
from typing import Optional, List, Dict, Union
import logging
import time

from src.api.kaggle_client import KaggleAPIClient
from src.api.kernels import KernelClient, KernelMetadata
from src.utils.helpers import timer, retry_on_exception

logger = logging.getLogger(__name__)

class KernelWorkflowManager:
    def __init__(self):
        """Initialize the kernel workflow manager"""
        self.kaggle_client = KaggleAPIClient()
        self.kernel_client = KernelClient(self.kaggle_client)
        self._load_configs()

    def _load_configs(self):
        """Load operational configurations"""
        try:
            with open('operational_configs/kernel_configs/runtime_settings.yaml', 'r') as f:
                self.runtime_config = yaml.safe_load(f)
            with open('operational_configs/kernel_configs/resource_limits.yaml', 'r') as f:
                self.resource_config = yaml.safe_load(f)
            logger.info("Successfully loaded kernel configurations")
        except Exception as e:
            logger.error(f"Error loading configurations: {str(e)}")
            raise

    @timer
    @retry_on_exception(retries=3, delay=1)
    def push_kernel(
        self,
        folder_path: Path,
        metadata: KernelMetadata,
        wait_for_completion: bool = True
    ) -> Dict:
        """
        Push a kernel to Kaggle

        Args:
            folder_path: Path to kernel files
            metadata: Kernel metadata
            wait_for_completion: Whether to wait for processing

        Returns:
            Response from the API
        """
        try:
            # Verify kernel files
            if not self._verify_kernel_files(folder_path, metadata.language):
                raise ValueError("Invalid kernel files structure")

            # Push kernel
            result = self.kernel_client.push_kernel(
                folder_path,
                metadata
            )

            if wait_for_completion:
                result = self.wait_for_kernel_completion(
                    metadata.title,
                    timeout=self.runtime_config.get('execution_timeout', 3600)
                )

            logger.info(f"Successfully pushed kernel: {result}")
            return result

        except Exception as e:
            logger.error(f"Error pushing kernel: {str(e)}")
            raise

    def _verify_kernel_files(self, folder_path: Path, language: str) -> bool:
        """Verify kernel files structure"""
        try:
            if language.lower() == 'python':
                main_file = folder_path / 'kernel.ipynb'
                if not main_file.exists():
                    main_file = folder_path / 'script.py'
            else:  # R
                main_file = folder_path / 'kernel.Rmd'
                if not main_file.exists():
                    main_file = folder_path / 'script.R'

            return main_file.exists()

        except Exception as e:
            logger.error(f"Error verifying kernel files: {str(e)}")
            return False

    @timer
    def pull_kernel(
        self,
        owner: str,
        kernel_name: str,
        version: Optional[str] = None,
        custom_path: Optional[Path] = None
    ) -> Path:
        """
        Pull a kernel from Kaggle

        Args:
            owner: Kernel owner's username
            kernel_name: Name of the kernel
            version: Specific version to pull
            custom_path: Optional custom download path

        Returns:
            Path to downloaded kernel
        """
        try:
            # Determine download path
            base_path = custom_path or Path(self.runtime_config['kernel_settings']['default_path'])
            download_path = base_path / f"{owner}_{kernel_name}"
            download_path.mkdir(parents=True, exist_ok=True)

            # Pull kernel
            kernel_path = self.kernel_client.pull_kernel(
                owner,
                kernel_name,
                version=version,
                path=download_path
            )

            logger.info(f"Successfully pulled kernel to {kernel_path}")
            return kernel_path

        except Exception as e:
            logger.error(f"Error pulling kernel: {str(e)}")
            raise

    def wait_for_kernel_completion(
        self,
        kernel_name: str,
        timeout: int = 3600,
        check_interval: int = 10
    ) -> Dict:
        """
        Wait for kernel execution to complete

        Args:
            kernel_name: Name of the kernel
            timeout: Maximum time to wait in seconds
            check_interval: Time between status checks

        Returns:
            Final kernel status
        """
        start_time = time.time()
        while True:
            try:
                status = self.kernel_client.get_kernel_status(kernel_name)

                if status.get('status') == 'complete':
                    result = self.kernel_client.get_kernel_output(kernel_name)
                    logger.info(f"Kernel {kernel_name} completed successfully")
                    return result

                if status.get('status') == 'error':
                    error_msg = status.get('errorMessage', 'Unknown error')
                    raise RuntimeError(f"Kernel execution failed: {error_msg}")

                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Kernel execution not completed after {timeout} seconds")

                time.sleep(check_interval)

            except Exception as e:
                logger.error(f"Error checking kernel status: {str(e)}")
                raise

    @timer
    def list_kernel_versions(
        self,
        owner: str,
        kernel_name: str
    ) -> List[Dict]:
        """
        List all versions of a kernel

        Args:
            owner: Kernel owner's username
            kernel_name: Name of the kernel

        Returns:
            List of kernel versions
        """
        try:
            versions = self.kernel_client.list_kernel_versions(
                owner,
                kernel_name
            )
            logger.info(f"Found {len(versions)} versions for kernel {kernel_name}")
            return versions

        except Exception as e:
            logger.error(f"Error listing kernel versions: {str(e)}")
            raise

def main():
    """Example usage of kernel workflows"""
    try:
        # Initialize manager
        manager = KernelWorkflowManager()

        # Example of pulling an existing kernel
        kernel_path = manager.pull_kernel(
            owner="username",
            kernel_name="example-kernel"
        )
        print(f"\nPulled kernel to: {kernel_path}")

        # Example of pushing a new kernel
        metadata = KernelMetadata(
            title="Example Kernel",
            language="python",
            kernel_type="notebook",
            is_private=False,
            enable_gpu=False,
            enable_internet=True,
            dataset_sources=["titanic"],
            competition_sources=None
        )

        new_kernel_path = Path("data/kernels/scripts/example_kernel")
        if new_kernel_path.exists():
            result = manager.push_kernel(
                new_kernel_path,
                metadata,
                wait_for_completion=True
            )
            print(f"\nPush result: {result}")

        # List kernel versions
        versions = manager.list_kernel_versions(
            owner="username",
            kernel_name="example-kernel"
        )
        print("\nKernel Versions:")
        for version in versions:
            print(f"Version: {version.get('version')}")

    except Exception as e:
        print(f"Error in kernel workflow: {str(e)}")

if __name__ == "__main__":
    main()
