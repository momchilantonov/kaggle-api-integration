from pathlib import Path
import yaml
from typing import Optional, List, Dict
import logging

from src.api.kaggle_client import KaggleAPIClient
from src.api.datasets import DatasetClient
from src.handlers.data_handlers import DataHandler
from src.utils.helpers import timer, retry_on_exception

logger = logging.getLogger(__name__)

class DatasetDownloadManager:
    def __init__(self):
        """Initialize the dataset download manager with necessary clients"""
        self.kaggle_client = KaggleAPIClient()
        self.dataset_client = DatasetClient(self.kaggle_client)
        self.data_handler = DataHandler()
        self._load_configs()

    def _load_configs(self):
        """Load operational configurations"""
        try:
            with open('operational_configs/dataset_configs/datasets.yaml', 'r') as f:
                self.dataset_config = yaml.safe_load(f)
            with open('operational_configs/dataset_configs/download_settings.yaml', 'r') as f:
                self.download_config = yaml.safe_load(f)
            logger.info("Successfully loaded dataset configurations")
        except Exception as e:
            logger.error(f"Error loading configurations: {str(e)}")
            raise

    @timer
    @retry_on_exception(retries=3, delay=1)
    def download_dataset(
        self,
        dataset_name: str,
        custom_path: Optional[Path] = None
    ) -> Path:
        """
        Download a dataset using predefined configurations

        Args:
            dataset_name: Name of the dataset from config
            custom_path: Optional custom download path

        Returns:
            Path to downloaded dataset
        """
        try:
            # Get dataset info from config
            dataset_info = self.dataset_config['frequently_used_datasets'].get(dataset_name)
            if not dataset_info:
                raise ValueError(f"Dataset {dataset_name} not found in configurations")

            # Determine download path
            base_path = custom_path or Path(self.download_config['download_preferences']['default_path'])
            download_path = base_path / dataset_name

            # Download dataset
            dataset_path = self.dataset_client.download_dataset(
                owner_slug=dataset_info['owner'],
                dataset_slug=dataset_info['dataset'],
                path=download_path,
                unzip=self.download_config['download_preferences']['auto_extract']
            )

            logger.info(f"Successfully downloaded dataset {dataset_name} to {dataset_path}")
            return dataset_path

        except Exception as e:
            logger.error(f"Error downloading dataset {dataset_name}: {str(e)}")
            raise

    @timer
    def process_downloaded_dataset(
        self,
        dataset_path: Path,
        process_config: Dict
    ) -> Path:
        """
        Process downloaded dataset according to configuration

        Args:
            dataset_path: Path to downloaded dataset
            process_config: Configuration for processing

        Returns:
            Path to processed dataset
        """
        try:
            # Create processed data directory
            processed_path = Path("data/datasets/processed") / dataset_path.name
            processed_path.mkdir(parents=True, exist_ok=True)

            # Process each file according to config
            for file_name in process_config['files']:
                input_file = dataset_path / file_name
                if not input_file.exists():
                    logger.warning(f"File {file_name} not found in dataset")
                    continue

                # Read and process file
                df = self.data_handler.read_csv(input_file)

                # Apply processing steps
                if process_config.get('handle_missing'):
                    df = self.data_handler.handle_missing_values(
                        df,
                        process_config['missing_strategy']
                    )

                # Save processed file
                output_file = processed_path / f"processed_{file_name}"
                self.data_handler.write_csv(df, output_file)
                logger.info(f"Processed {file_name} saved to {output_file}")

            return processed_path

        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            raise

    def get_dataset_stats(self, dataset_path: Path) -> Dict:
        """
        Get statistics for downloaded dataset

        Args:
            dataset_path: Path to dataset

        Returns:
            Dictionary with dataset statistics
        """
        try:
            stats = {}
            for file_path in dataset_path.glob('*.csv'):
                df = self.data_handler.read_csv(file_path)
                stats[file_path.name] = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'missing_values': df.isnull().sum().sum(),
                    'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
                    'column_stats': self.data_handler.calculate_basic_stats(df)
                }
            return stats
        except Exception as e:
            logger.error(f"Error calculating dataset statistics: {str(e)}")
            raise

def main():
    """Example usage of dataset download workflows"""
    try:
        # Initialize manager
        manager = DatasetDownloadManager()

        # Download Titanic dataset
        dataset_path = manager.download_dataset("titanic")
        print(f"\nDownloaded Titanic dataset to: {dataset_path}")

        # Process dataset
        process_config = {
            'files': ['train.csv', 'test.csv'],
            'handle_missing': True,
            'missing_strategy': {
                'Age': 'mean',
                'Cabin': 'drop',
                'Embarked': 'mode'
            }
        }
        processed_path = manager.process_downloaded_dataset(dataset_path, process_config)
        print(f"\nProcessed dataset saved to: {processed_path}")

        # Get dataset statistics
        stats = manager.get_dataset_stats(dataset_path)
        print("\nDataset Statistics:")
        for file_name, file_stats in stats.items():
            print(f"\n{file_name}:")
            print(f"  Rows: {file_stats['rows']}")
            print(f"  Columns: {file_stats['columns']}")
            print(f"  Missing Values: {file_stats['missing_values']}")
            print(f"  Memory Usage: {file_stats['memory_usage']:.2f} MB")

    except Exception as e:
        print(f"Error in dataset workflow: {str(e)}")

if __name__ == "__main__":
    main()
