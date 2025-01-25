from pathlib import Path
import yaml
from typing import Optional, List, Dict, Union
import logging
import shutil
import json
from datetime import datetime

from src.api.kaggle_client import KaggleAPIClient
from src.api.kernels import KernelClient, KernelMetadata
from src.utils.helpers import timer, retry_on_exception

logger = logging.getLogger(__name__)

class KernelManagementManager:
    def __init__(self):
        """Initialize the kernel management manager"""
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
    def version_kernel(
        self,
        kernel_path: Union[str, Path],
        version_notes: str,
        metadata: Optional[KernelMetadata] = None
    ) -> Dict:
        """
        Create new version of kernel

        Args:
            kernel_path: Path to kernel files
            version_notes: Notes for version
            metadata: Optional updated metadata

        Returns:
            Version creation response
        """
        try:
            kernel_path = Path(kernel_path)
            if not kernel_path.exists():
                raise FileNotFoundError(f"Kernel path not found: {kernel_path}")

            # Create version directory
            version_dir = self._prepare_version_directory(kernel_path)

            # Update metadata if provided
            if metadata:
                self._update_kernel_metadata(version_dir, metadata)

            # Push new version
            result = self.kernel_client.push_kernel(
                version_dir,
                metadata or self._load_existing_metadata(kernel_path),
                version_notes=version_notes
            )

            logger.info(f"Created new kernel version: {result}")
            return result

        except Exception as e:
            logger.error(f"Error creating kernel version: {str(e)}")
            raise

    def _prepare_version_directory(self, kernel_path: Path) -> Path:
        """Prepare directory for new version"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version_dir = kernel_path.parent / f"{kernel_path.name}_version_{timestamp}"
            version_dir.mkdir(parents=True, exist_ok=True)

            # Copy kernel files
            for file_path in kernel_path.glob('*'):
                if file_path.is_file():
                    shutil.copy2(file_path, version_dir)

            return version_dir

        except Exception as e:
            logger.error(f"Error preparing version directory: {str(e)}")
            raise

    def _load_existing_metadata(self, kernel_path: Path) -> KernelMetadata:
        """Load existing kernel metadata"""
        try:
            metadata_path = kernel_path / 'kernel-metadata.json'
            if not metadata_path.exists():
                raise FileNotFoundError("Kernel metadata file not found")

            with open(metadata_path) as f:
                metadata_dict = json.load(f)

            return KernelMetadata(**metadata_dict)

        except Exception as e:
            logger.error(f"Error loading kernel metadata: {str(e)}")
            raise

    def _update_kernel_metadata(
        self,
        kernel_dir: Path,
        metadata: KernelMetadata
    ) -> None:
        """Update kernel metadata file"""
        try:
            metadata_path = kernel_dir / 'kernel-metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)

        except Exception as e:
            logger.error(f"Error updating kernel metadata: {str(e)}")
            raise

    @timer
    def manage_kernel_resources(
        self,
        kernel_name: str,
        resource_config: Dict
    ) -> Dict:
        """
        Update kernel resource configuration

        Args:
            kernel_name: Name of the kernel
            resource_config: Resource configuration

        Returns:
            Updated kernel status
        """
        try:
            # Get current kernel metadata
            metadata = self.kernel_client.get_kernel_metadata(kernel_name)

            # Update resource settings
            metadata.enable_gpu = resource_config.get('enable_gpu', False)
            metadata.enable_internet = resource_config.get('enable_internet', False)

            # Update kernel with new settings
            result = self.kernel_client.update_kernel(
                kernel_name,
                metadata
            )

            logger.info(f"Updated kernel resources: {result}")
            return result

        except Exception as e:
            logger.error(f"Error managing kernel resources: {str(e)}")
            raise

    @timer
    def batch_update_kernels(
        self,
        kernel_configs: List[Dict]
    ) -> List[Dict]:
        """
        Update multiple kernels in batch

        Args:
            kernel_configs: List of kernel configurations

        Returns:
            List of update results
        """
        try:
            results = []
            for config in kernel_configs:
                try:
                    result = self.manage_kernel_resources(
                        config['kernel_name'],
                        config['resources']
                    )
                    results.append({
                        'kernel': config['kernel_name'],
                        'status': 'success',
                        'result': result
                    })
                except Exception as e:
                    results.append({
                        'kernel': config['kernel_name'],
                        'status': 'error',
                        'error': str(e)
                    })

            return results

        except Exception as e:
            logger.error(f"Error in batch update: {str(e)}")
            raise

    @timer
    def create_kernel_backup(
        self,
        kernel_name: str,
        include_data: bool = True
    ) -> Path:
        """
        Create backup of kernel and its data

        Args:
            kernel_name: Name of the kernel
            include_data: Whether to include data files

        Returns:
            Path to backup directory
        """
        try:
            # Create backup directory
            backup_dir = Path(self.runtime_config['backup_settings']['path'])
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            kernel_backup_dir = backup_dir / f"{kernel_name}_backup_{timestamp}"
            kernel_backup_dir.mkdir(parents=True, exist_ok=True)

            # Pull kernel files
            self.kernel_client.pull_kernel(
                kernel_name,
                path=kernel_backup_dir
            )

            # Include data files if requested
            if include_data:
                data_sources = self._get_kernel_data_sources(kernel_name)
                for source in data_sources:
                    try:
                        self.kernel_client.download_kernel_data(
                            source,
                            kernel_backup_dir / 'data'
                        )
                    except Exception as e:
                        logger.warning(f"Error backing up data source {source}: {str(e)}")

            logger.info(f"Created kernel backup at {kernel_backup_dir}")
            return kernel_backup_dir

        except Exception as e:
            logger.error(f"Error creating kernel backup: {str(e)}")
            raise

    def _get_kernel_data_sources(self, kernel_name: str) -> List[str]:
        """Get data sources used by kernel"""
        try:
            metadata = self.kernel_client.get_kernel_metadata(kernel_name)
            return (metadata.dataset_sources or []) + (metadata.competition_sources or [])

        except Exception as e:
            logger.error(f"Error getting kernel data sources: {str(e)}")
            return []

def main():
    """Example usage of kernel management operations"""
    try:
        # Initialize manager
        manager = KernelManagementManager()

        # Example: Create new kernel version
        kernel_path = Path("data/kernels/example_kernel")
        if kernel_path.exists():
            result = manager.version_kernel(
                kernel_path,
                version_notes="Updated data processing"
            )
            print(f"\nCreated new version: {result}")

        # Example: Update kernel resources
        resource_config = {
            'enable_gpu': True,
            'enable_internet': True
        }
        result = manager.manage_kernel_resources(
            "example-kernel",
            resource_config
        )
        print(f"\nUpdated resources: {result}")

        # Example: Create kernel backup
        backup_path = manager.create_kernel_backup(
            "example-kernel",
            include_data=True
        )
        print(f"\nCreated backup at: {backup_path}")

    except Exception as e:
        print(f"Error in kernel management operations: {str(e)}")

if __name__ == "__main__":
    main()
