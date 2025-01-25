from pathlib import Path
import yaml
from typing import Optional, List, Dict, Union
import logging
import shutil
import json

from src.api.kaggle_client import KaggleAPIClient
from src.api.datasets import DatasetClient, DatasetMetadata
from src.handlers.data_handlers import DataHandler
from src.utils.helpers import timer, retry_on_exception, compress_file

logger = logging.getLogger(__name__)

class DatasetUploadManager:
    def __init__(self):
        """Initialize the dataset upload manager"""
        self.kaggle_client = KaggleAPIClient()
        self.dataset_client = DatasetClient(self.kaggle_client)
        self.data_handler = DataHandler()
        self._load_configs()

    def _load_configs(self):
        """Load operational configurations"""
        try:
            with open('operational_configs/dataset_configs/datasets.yaml', 'r') as f:
                self.dataset_config = yaml.safe_load(f)
            logger.info("Successfully loaded dataset configurations")
        except Exception as e:
            logger.error(f"Error loading configurations: {str(e)}")
            raise

    @timer
    def prepare_dataset_folder(
        self,
        source_files: List[Union[str, Path]],
        dataset_name: str,
        include_metadata: bool = True
    ) -> Path:
        """
        Prepare a folder with dataset files and metadata

        Args:
            source_files: List of files to include
            dataset_name: Name for the dataset
            include_metadata: Whether to include dataset-metadata.json

        Returns:
            Path to prepared folder
        """
        try:
            # Create dataset directory
            dataset_dir = Path("data/datasets/uploads") / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)

            # Copy files to dataset directory
            for file_path in source_files:
                file_path = Path(file_path)
                if not file_path.exists():
                    raise FileNotFoundError(f"File not found: {file_path}")

                # Copy file
                shutil.copy2(file_path, dataset_dir / file_path.name)
                logger.info(f"Copied {file_path.name} to dataset directory")

            # Create metadata file if requested
            if include_metadata:
                self._create_metadata_file(dataset_dir, dataset_name)

            return dataset_dir

        except Exception as e:
            logger.error(f"Error preparing dataset folder: {str(e)}")
            raise

    def _create_metadata_file(
        self,
        dataset_dir: Path,
        dataset_name: str
    ) -> Path:
        """Create dataset metadata file"""
        try:
            metadata = {
                "title": dataset_name,
                "id": f"{self.kaggle_client.credentials['username']}/{dataset_name}",
                "licenses": [{"name": "CC0-1.0"}],
                "keywords": [],
                "resources": []
            }

            # Add resources information
            for file_path in dataset_dir.glob('*'):
                if file_path.name != 'dataset-metadata.json':
                    metadata['resources'].append({
                        "path": file_path.name,
                        "description": f"File: {file_path.name}"
                    })

            metadata_path = dataset_dir / 'dataset-metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Created metadata file: {metadata_path}")
            return metadata_path

        except Exception as e:
            logger.error(f"Error creating metadata file: {str(e)}")
            raise

    @timer
    @retry_on_exception(retries=3, delay=1)
    def upload_dataset(
        self,
        dataset_dir: Path,
        metadata: DatasetMetadata,
        public: bool = True
    ) -> Dict:
        """
        Upload dataset to Kaggle

        Args:
            dataset_dir: Directory containing dataset files
            metadata: Dataset metadata
            public: Whether dataset should be public

        Returns:
            Upload response
        """
        try:
            result = self.dataset_client.create_dataset(
                folder_path=dataset_dir,
                metadata=metadata,
                public=public
            )
            logger.info(f"Successfully uploaded dataset: {result}")
            return result

        except Exception as e:
            logger.error(f"Error uploading dataset: {str(e)}")
            raise

    @timer
    def create_dataset_version(
        self,
        dataset_dir: Path,
        version_notes: str,
        delete_old_versions: bool = False
    ) -> Dict:
        """
        Create new version of existing dataset

        Args:
            dataset_dir: Directory containing updated files
            version_notes: Notes describing changes
            delete_old_versions: Whether to delete previous versions

        Returns:
            Version creation response
        """
        try:
            result = self.dataset_client.create_version(
                folder_path=dataset_dir,
                version_notes=version_notes,
                delete_old_versions=delete_old_versions
            )
            logger.info(f"Successfully created dataset version: {result}")
            return result

        except Exception as e:
            logger.error(f"Error creating dataset version: {str(e)}")
            raise

    @timer
    def update_dataset_metadata(
        self,
        dataset_slug: str,
        metadata: DatasetMetadata
    ) -> Dict:
        """
        Update dataset metadata

        Args:
            dataset_slug: Dataset identifier
            metadata: Updated metadata

        Returns:
            Update response
        """
        try:
            owner_slug = self.kaggle_client.credentials['username']
            result = self.dataset_client.update_metadata(
                owner_slug=owner_slug,
                dataset_slug=dataset_slug,
                metadata=metadata
            )
            logger.info(f"Successfully updated dataset metadata: {result}")
            return result

        except Exception as e:
            logger.error(f"Error updating dataset metadata: {str(e)}")
            raise

    def prepare_files_for_upload(
        self,
        source_files: List[Union[str, Path]],
        compress: bool = True
    ) -> List[Path]:
        """
        Prepare files for upload (compress if needed)

        Args:
            source_files: List of files to prepare
            compress: Whether to compress large files

        Returns:
            List of prepared file paths
        """
        try:
            prepared_files = []
            for file_path in source_files:
                file_path = Path(file_path)
                if not file_path.exists():
                    raise FileNotFoundError(f"File not found: {file_path}")

                # Check if compression needed
                if compress and file_path.stat().st_size > 100 * 1024 * 1024:  # 100MB
                    compressed_path = compress_file(file_path, method='gzip')
                    prepared_files.append(compressed_path)
                    logger.info(f"Compressed {file_path.name}")
                else:
                    prepared_files.append(file_path)

            return prepared_files

        except Exception as e:
            logger.error(f"Error preparing files: {str(e)}")
            raise

def main():
    """Example usage of dataset upload operations"""
    try:
        # Initialize manager
        manager = DatasetUploadManager()

        # Prepare files for upload
        source_files = [
            Path("data/processed/train.csv"),
            Path("data/processed/test.csv"),
            Path("data/processed/sample_submission.csv")
        ]

        # Prepare dataset folder
        dataset_name = "example-dataset"
        dataset_dir = manager.prepare_dataset_folder(
            source_files,
            dataset_name,
            include_metadata=True
        )
        print(f"\nPrepared dataset directory: {dataset_dir}")

        # Create metadata
        metadata = DatasetMetadata(
            title="Example Dataset",
            slug="example-dataset",
            description="This is an example dataset",
            licenses=[{"name": "CC0-1.0"}],
            keywords=["example", "test"],
            collaborators=None
        )

        # Upload dataset
        result = manager.upload_dataset(
            dataset_dir,
            metadata,
            public=True
        )
        print(f"\nUpload result: {result}")

        # Create new version
        version_result = manager.create_dataset_version(
            dataset_dir,
            "Updated data files",
            delete_old_versions=False
        )
        print(f"\nVersion creation result: {version_result}")

    except Exception as e:
        print(f"Error in dataset upload operations: {str(e)}")

if __name__ == "__main__":
    main()
