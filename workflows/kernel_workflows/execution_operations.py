from pathlib import Path
from typing import Dict, Optional, Union, List
import logging
from datetime import datetime
import shutil
import json
import time

from src.api.kernels import KernelClient, KernelMetadata
from src.utils.path_manager import PathManager
from src.utils.error_handlers import handle_api_errors, retry_with_backoff

logger = logging.getLogger(__name__)

class KernelWorkflowManager:
    """Manages kernel execution workflows"""

    def __init__(self):
        self.kernel_client = KernelClient()
        self.path_manager = PathManager()
        # Ensure required directories exist
        self.path_manager.ensure_directories()

    @handle_api_errors
    def push_kernel(
        self,
        folder_path: Union[str, Path],
        metadata: KernelMetadata,
        wait_for_completion: bool = True,
        timeout: int = 3600
    ) -> Dict:
        """Push and optionally execute a kernel"""
        try:
            folder_path = Path(folder_path)
            if not folder_path.exists():
                raise FileNotFoundError(f"Kernel folder not found: {folder_path}")

            # Validate kernel files
            self._validate_kernel_files(folder_path, metadata.language)

            # Create backup before push
            backup_dir = self.path_manager.get_path('kernels', 'scripts') / 'backups'
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / f"{folder_path.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copytree(folder_path, backup_path)

            # Push kernel
            result = self.kernel_client.push_kernel(folder_path, metadata)

            # Wait for execution if requested
            if wait_for_completion:
                result = self.wait_for_kernel_completion(
                    metadata.title,
                    timeout=timeout
                )

            # Log push details
            push_log_path = self.path_manager.get_path('kernels', 'scripts') / 'kernels.log'
            with open(push_log_path, 'a') as f:
                f.write(
                    f"{datetime.now()},{metadata.title},{result.get('id', 'N/A')},"
                    f"{result.get('status', 'N/A')}\n"
                )

            return result

        except Exception as e:
            logger.error(f"Error pushing kernel: {str(e)}")
            raise

    def _validate_kernel_files(self, folder_path: Path, language: str) -> None:
        """Validate kernel files structure"""
        if language.lower() == 'python':
            main_file = folder_path / 'kernel.ipynb'
            if not main_file.exists():
                main_file = folder_path / 'script.py'
        else:  # R
            main_file = folder_path / 'kernel.Rmd'
            if not main_file.exists():
                main_file = folder_path / 'script.R'

        if not main_file.exists():
            raise ValueError(
                f"No valid kernel file found for language {language}. "
                f"Expected: kernel.ipynb/script.py for Python or kernel.Rmd/script.R for R"
            )

    @handle_api_errors
    def pull_kernel(
        self,
        owner: str,
        kernel_name: str,
        version: Optional[str] = None,
        custom_path: Optional[Path] = None
    ) -> Path:
        """Pull a kernel and save it locally"""
        try:
            # Get kernel path
            kernel_path = (
                custom_path or
                self.path_manager.get_path('kernels', 'scripts') / f"{owner}_{kernel_name}"
            )
            kernel_path.mkdir(parents=True, exist_ok=True)

            # Pull kernel
            kernel_path = self.kernel_client.pull_kernel(
                owner,
                kernel_name,
                version=version,
                path=kernel_path
            )

            # Create metadata about pull
            metadata = {
                'owner': owner,
                'kernel_name': kernel_name,
                'version': version,
                'pull_date': datetime.now().isoformat(),
                'files': [
                    {
                        'name': f.name,
                        'size': f.stat().st_size,
                        'type': f.suffix[1:] if f.suffix else 'unknown'
                    }
                    for f in kernel_path.glob('*.*')
                ]
            }

            metadata_path = kernel_path / 'pull_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            return kernel_path

        except Exception as e:
            logger.error(f"Error pulling kernel: {str(e)}")
            raise

    @retry_with_backoff(max_retries=3)
    def wait_for_kernel_completion(
        self,
        kernel_name: str,
        timeout: int = 3600,
        check_interval: int = 10
    ) -> Dict:
        """Wait for kernel execution to complete"""
        try:
            start_time = datetime.now()
            while True:
                # Check status
                status = self.kernel_client.get_kernel_status(kernel_name)

                if status.get('status') == 'complete':
                    # Get kernel output
                    output = self.kernel_client.get_kernel_output(kernel_name)

                    # Save output
                    output_path = (
                        self.path_manager.get_path('kernels', 'outputs') /
                        f"{kernel_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    )
                    with open(output_path, 'w') as f:
                        json.dump(output, f, indent=2)

                    return {'status': 'complete', 'output_path': output_path}

                if status.get('status') == 'error':
                    error_msg = status.get('error', 'Unknown error')
                    raise RuntimeError(f"Kernel execution failed: {error_msg}")

                # Check timeout
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > timeout:
                    raise TimeoutError(
                        f"Kernel execution not completed after {timeout} seconds"
                    )

                time.sleep(check_interval)

        except Exception as e:
            logger.error(f"Error waiting for kernel completion: {str(e)}")
            raise

    @handle_api_errors
    def get_kernel_status(
        self,
        owner: str,
        kernel_name: str,
        version: Optional[str] = None
    ) -> Dict:
        """Get detailed kernel status"""
        try:
            # Get basic status
            status = self.kernel_client.get_kernel_status(
                owner,
                kernel_name,
                version
            )

            # Get version history
            versions = self.kernel_client.list_kernel_versions(owner, kernel_name)

            # Enhance status information
            enhanced_status = {
                'status': status.get('status'),
                'last_run': status.get('lastRunTime'),
                'total_versions': len(versions),
                'latest_version': versions[0] if versions else None,
                'all_versions': versions
            }

            return enhanced_status

        except Exception as e:
            logger.error(f"Error getting kernel status: {str(e)}")
            raise

if __name__ == '__main__':
    # Example usage
    manager = KernelWorkflowManager()

    try:
        # Push a kernel
        kernel_path = Path("path/to/kernel")
        metadata = KernelMetadata(
            title="Example Kernel",
            language="python",
            kernel_type="script"
        )
        result = manager.push_kernel(kernel_path, metadata)
        print(f"Push result: {result}")

        # Pull a kernel
        kernel_path = manager.pull_kernel("owner", "kernel-name")
        print(f"Pulled kernel to: {kernel_path}")

        # Get kernel status
        status = manager.get_kernel_status("owner", "kernel-name")
        print(f"Kernel status: {status}")

    except Exception as e:
        print(f"Error: {str(e)}")
