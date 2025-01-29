from pathlib import Path
from typing import Dict, Optional, Union, List
import logging
from datetime import datetime
import shutil
import json
import time

from src.api.kernels import KernelClient, KernelMetadata
from src.utils.path_manager import PathManager
from src.utils.error_handlers import handle_api_errors

logger = logging.getLogger(__name__)

class KernelManagementManager:
    """Manages kernel management workflows"""

    def __init__(self):
        self.kernel_client = KernelClient()
        self.path_manager = PathManager()
        # Ensure required directories exist
        self.path_manager.ensure_directories()

    @handle_api_errors
    def version_kernel(
        self,
        kernel_path: Union[str, Path],
        version_notes: str,
        metadata: Optional[KernelMetadata] = None
    ) -> Dict:
        """Create new version of kernel"""
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

            # Log version creation
            version_log_path = self.path_manager.get_path('kernels', 'scripts') / 'versions.log'
            with open(version_log_path, 'a') as f:
                f.write(
                    f"{datetime.now()},{kernel_path.name},{result.get('version', 'N/A')},"
                    f"{version_notes}\n"
                )

            return result

        except Exception as e:
            logger.error(f"Error creating kernel version: {str(e)}")
            raise

    def _prepare_version_directory(self, kernel_path: Path) -> Path:
        """Prepare directory for new version"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version_dir = (
                self.path_manager.get_path('kernels', 'scripts') /
                'versions' /
                f"{kernel_path.name}_version_{timestamp}"
            )
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

    def _update_kernel_metadata(self, kernel_dir: Path, metadata: KernelMetadata) -> None:
        """Update kernel metadata file"""
        try:
            metadata_path = kernel_dir / 'kernel-metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)

        except Exception as e:
            logger.error(f"Error updating kernel metadata: {str(e)}")
            raise

    @handle_api_errors
    def manage_kernel_resources(
        self,
        kernel_name: str,
        resource_config: Dict
    ) -> Dict:
        """Update kernel resource configuration"""
        try:
            # Get current kernel metadata
            metadata = self.kernel_client.get_kernel_metadata(kernel_name)

            # Update resource settings
            metadata.enable_gpu = resource_config.get('enable_gpu', False)
            metadata.enable_internet = resource_config.get('enable_internet', False)

            # Additional resource configurations
            if 'memory_limit' in resource_config:
                metadata.memory_limit = resource_config['memory_limit']
            if 'timeout' in resource_config:
                metadata.timeout = resource_config['timeout']

            # Update kernel with new settings
            result = self.kernel_client.update_kernel(kernel_name, metadata)

            # Log resource update
            resource_log_path = self.path_manager.get_path('kernels', 'scripts') / 'resources.log'
            with open(resource_log_path, 'a') as f:
                f.write(
                    f"{datetime.now()},{kernel_name},Resource Update,"
                    f"{json.dumps(resource_config)}\n"
                )

            return result

        except Exception as e:
            logger.error(f"Error managing kernel resources: {str(e)}")
            raise

    @handle_api_errors
    def create_kernel_backup(
        self,
        kernel_name: str,
        include_data: bool = True
    ) -> Path:
        """Create backup of kernel and its data"""
        try:
            # Create backup directory
            backup_dir = (
                self.path_manager.get_path('kernels', 'scripts') /
                'backups' /
                datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Pull kernel files
            self.kernel_client.pull_kernel(
                kernel_name,
                path=backup_dir
            )

            # Include data files if requested
            if include_data:
                data_sources = self._get_kernel_data_sources(kernel_name)
                for source in data_sources:
                    try:
                        data_dir = backup_dir / 'data'
                        data_dir.mkdir(exist_ok=True)
                        self.kernel_client.download_kernel_data(
                            source,
                            data_dir
                        )
                    except Exception as e:
                        logger.warning(f"Error backing up data source {source}: {str(e)}")

            # Create backup metadata
            metadata = {
                'kernel_name': kernel_name,
                'backup_date': datetime.now().isoformat(),
                'include_data': include_data,
                'data_sources': data_sources if include_data else [],
                'files': [
                    {
                        'name': f.name,
                        'size': f.stat().st_size,
                        'type': f.suffix[1:] if f.suffix else 'unknown'
                    }
                    for f in backup_dir.glob('**/*')
                ]
            }

            metadata_path = backup_dir / 'backup_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            return backup_dir

        except Exception as e:
            logger.error(f"Error creating kernel backup: {str(e)}")
            raise

    def _get_kernel_data_sources(self, kernel_name: str) -> List[str]:
        """Get data sources used by kernel"""
        try:
            metadata = self.kernel_client.get_kernel_metadata(kernel_name)
            sources = []
            if metadata.dataset_sources:
                sources.extend(metadata.dataset_sources)
            if metadata.competition_sources:
                sources.extend(metadata.competition_sources)
            return sources

        except Exception as e:
            logger.error(f"Error getting kernel data sources: {str(e)}")
            return []

if __name__ == '__main__':
    # Example usage
    manager = KernelManagementManager()

    try:
        # Create kernel version
        kernel_path = Path("path/to/kernel")
        result = manager.version_kernel(
            kernel_path,
            "Updated data processing"
        )
        print(f"Created new version: {result}")

        # Update kernel resources
        resource_config = {
            'enable_gpu': True,
            'enable_internet': True,
            'memory_limit': '16g'
        }
        result = manager.manage_kernel_resources(
            "example-kernel",
            resource_config
        )
        print(f"Updated resources: {result}")

        # Create kernel backup
        backup_path = manager.create_kernel_backup(
            "example-kernel",
            include_data=True
        )
        print(f"Created backup at: {backup_path}")

    except Exception as e:
        print(f"Error: {str(e)}")
