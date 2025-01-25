from typing import Dict, List, Optional, Union
from pathlib import Path
import time
from dataclasses import dataclass

from .kaggle_client import KaggleAPIClient
from config.settings import setup_logger

logger = setup_logger('kaggle_kernels', 'kaggle_kernels.log')

@dataclass
class KernelMetadata:
    """Kernel metadata for creation/update operations"""
    title: str
    language: str  # python or r
    kernel_type: str  # script or notebook
    is_private: bool = False
    enable_gpu: bool = False
    enable_internet: bool = False
    dataset_sources: Optional[List[str]] = None
    competition_sources: Optional[List[str]] = None
    kernel_sources: Optional[List[str]] = None

    def to_dict(self) -> Dict:
        """Convert metadata to dictionary format"""
        return {k: v for k, v in self.__dict__.items() if v is not None}

class KernelClient:
    """Client for handling Kaggle kernel operations"""

    def __init__(self, client: KaggleAPIClient):
        """Initialize with a KaggleAPIClient instance"""
        self.client = client

    def list_kernels(
        self,
        owner: Optional[str] = None,
        search: Optional[str] = None,
        language: Optional[str] = None,
        kernel_type: Optional[str] = None,
        output_type: Optional[str] = None,
        sort_by: Optional[str] = None,
        page: int = 1,
        page_size: int = 20
    ) -> List[Dict]:
        """
        List kernels on Kaggle with filtering options

        Args:
            owner: Filter by kernel owner
            search: Search terms
            language: Filter by language (python or r)
            kernel_type: Filter by type (script or notebook)
            output_type: Filter by output type
            sort_by: Sort results (hotness, votes, created, etc)
            page: Page number for pagination
            page_size: Number of results per page

        Returns:
            List of kernels matching criteria
        """
        params = {
            'owner': owner,
            'search': search,
            'language': language,
            'kernelType': kernel_type,
            'outputType': output_type,
            'sortBy': sort_by,
            'page': page,
            'pageSize': page_size
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        response = self.client.get('kernels_list', params=params)
        kernels = response.json()
        logger.info(f"Found {len(kernels)} kernels matching criteria")
        return kernels

    def push_kernel(
        self,
        folder_path: Union[str, Path],
        metadata: KernelMetadata
    ) -> Dict:
        """
        Push a new kernel or update existing one

        Args:
            folder_path: Path to kernel files
            metadata: Kernel metadata

        Returns:
            Response from the API
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Path not found: {folder_path}")

        if not folder_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {folder_path}")

        # First, push kernel metadata
        response = self.client.post(
            'kernel_push',
            json=metadata.to_dict()
        )

        # Then upload each file
        for file_path in folder_path.rglob('*'):
            if file_path.is_file():
                relative_path = str(file_path.relative_to(folder_path))
                self.client.upload_file(
                    'kernel_push',
                    file_path,
                    {
                        'path': relative_path
                    }
                )

        return response.json()

    def pull_kernel(
        self,
        owner: str,
        kernel_name: str,
        version: Optional[str] = None,
        path: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Download a kernel

        Args:
            owner: Kernel owner's username
            kernel_name: Name of the kernel
            version: Specific version to pull (latest if None)
            path: Path to save the kernel

        Returns:
            Path to the downloaded kernel
        """
        path = Path(path) if path else Path.cwd()
        path.mkdir(parents=True, exist_ok=True)

        params = {
            'ownerSlug': owner,
            'kernelSlug': kernel_name
        }
        if version:
            params['version'] = version

        response = self.client.get(
            'kernel_pull',
            params=params
        )

        kernel_path = path / kernel_name
        kernel_path.mkdir(exist_ok=True)

        # Save each file from the response
        kernel_files = response.json()
        for file_info in kernel_files:
            file_path = kernel_path / file_info['path']
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(file_info['source'])

        return kernel_path

    def get_kernel_status(
        self,
        owner: str,
        kernel_name: str,
        version: Optional[str] = None
    ) -> Dict:
        """
        Get kernel status

        Args:
            owner: Kernel owner's username
            kernel_name: Name of the kernel
            version: Specific version to check

        Returns:
            Kernel status information
        """
        params = {
            'ownerSlug': owner,
            'kernelSlug': kernel_name
        }
        if version:
            params['version'] = version

        response = self.client.get('kernel_status', params=params)
        return response.json()

    def wait_for_kernel_output(
        self,
        owner: str,
        kernel_name: str,
        timeout: int = 3600,  # 1 hour
        check_interval: int = 10
    ) -> Dict:
        """
        Wait for kernel execution output

        Args:
            owner: Kernel owner's username
            kernel_name: Name of the kernel
            timeout: Maximum time to wait in seconds
            check_interval: Time between checks in seconds

        Returns:
            Kernel output information

        Raises:
            TimeoutError: If kernel doesn't complete within timeout
            RuntimeError: If kernel execution fails
        """
        start_time = time.time()

        while True:
            status = self.get_kernel_status(owner, kernel_name)

            if status.get('status') == 'complete':
                response = self.client.get(
                    'kernel_output',
                    params={
                        'ownerSlug': owner,
                        'kernelSlug': kernel_name
                    }
                )
                logger.info(f"Kernel {owner}/{kernel_name} completed")
                return response.json()

            if status.get('status') == 'error':
                error_msg = status.get('errorMessage', 'Unknown error')
                logger.error(f"Kernel failed: {error_msg}")
                raise RuntimeError(f"Kernel failed: {error_msg}")

            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Kernel not completed after {timeout} seconds"
                )

            time.sleep(check_interval)

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
        response = self.client.get(
            'kernel_versions',
            params={
                'ownerSlug': owner,
                'kernelSlug': kernel_name
            }
        )
        versions = response.json()
        logger.info(f"Found {len(versions)} versions for kernel {owner}/{kernel_name}")
        return versions
