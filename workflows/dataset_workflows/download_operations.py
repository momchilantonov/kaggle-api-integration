from pathlib import Path
from typing import Dict, Optional, Union
import logging
from datetime import datetime
import pandas as pd

from src.api.datasets import DatasetClient
from src.handlers.data_handlers import DataHandler
from src.utils.path_manager import PathManager
from src.utils.error_handlers import handle_api_errors

logger = logging.getLogger(__name__)

class DatasetDownloadManager:
    """Manages dataset download and processing workflows"""

    def __init__(self, dataset_client):
        self.dataset_client = dataset_client
        self.data_handler = DataHandler()
        self.path_manager = PathManager()
        # # Ensure required directories exist
        self.path_manager.ensure_directories()

    @handle_api_errors
    def download_dataset(
        self,
        dataset_name: str,
        custom_path: Optional[Path] = None,
        unzip: bool = True
    ) -> Path:
        """Download and prepare dataset"""
        try:
            # Get dataset path
            dataset_path = (
                custom_path or
                self.path_manager.get_path('datasets', 'raw') / dataset_name
            )
            dataset_path.mkdir(parents=True, exist_ok=True)

            # Get owner and dataset slug from name
            if '/' in dataset_name:
                owner_slug, dataset_slug = dataset_name.split('/')
            else:
                # Use default owner if not specified
                owner_slug = 'kaggle'
                dataset_slug = dataset_name

            # Download dataset
            dataset_path = self.dataset_client.download_dataset(
                owner_slug=owner_slug,
                dataset_slug=dataset_slug,
                path=dataset_path,
                unzip=unzip
            )

            # Create metadata log
            metadata_path = dataset_path / 'dataset_metadata.json'
            if not metadata_path.exists():
                metadata = {
                    'name': dataset_name,
                    'download_date': datetime.now().isoformat(),
                    'files': [
                        {
                            'name': f.name,
                            'size': f.stat().st_size,
                            'type': f.suffix[1:] if f.suffix else 'unknown'
                        }
                        for f in dataset_path.glob('*.*')
                    ]
                }
                self.data_handler.write_json(metadata_path, metadata)

            logger.info(f"Downloaded dataset to {dataset_path}")
            return dataset_path

        except Exception as e:
            logger.error(f"Error downloading dataset: {str(e)}")
            raise

    @handle_api_errors
    def process_dataset(
        self,
        dataset_path: Path,
        process_config: Dict
    ) -> Path:
        """Process downloaded dataset"""
        try:
            # Get processed data path
            processed_path = (
                self.path_manager.get_path('datasets', 'processed') /
                dataset_path.name
            )
            processed_path.mkdir(parents=True, exist_ok=True)

            # Process each CSV file
            csv_files = list(dataset_path.glob('*.csv'))
            if not csv_files:
                raise ValueError(f"No CSV files found in {dataset_path}")

            processing_stats = []
            for file_path in csv_files:
                # Read data
                df = self.data_handler.read_csv(file_path)

                # Apply processing steps
                if process_config.get('handle_missing'):
                    df = self.data_handler.handle_missing_values(
                        df,
                        process_config['missing_strategy']
                    )

                if process_config.get('convert_types'):
                    df = self.data_handler.convert_dtypes(
                        df,
                        process_config['type_mapping']
                    )

                # Save processed file
                output_path = processed_path / f"processed_{file_path.name}"
                self.data_handler.write_csv(df, output_path)

                # Collect statistics
                stats = self.data_handler.calculate_basic_stats(df)
                processing_stats.append({
                    'file': file_path.name,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'stats': stats
                })

            # Save processing metadata
            metadata = {
                'original_dataset': dataset_path.name,
                'processing_date': datetime.now().isoformat(),
                'config': process_config,
                'statistics': processing_stats
            }
            self.data_handler.write_json(
                processed_path / 'processing_metadata.json',
                metadata
            )

            logger.info(f"Processed dataset saved to {processed_path}")
            return processed_path

        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            raise

    def get_dataset_stats(self, dataset_path: Path) -> Dict:
        """Get comprehensive dataset statistics"""
        try:
            stats = {}
            total_size = 0
            total_rows = 0

            # Analyze each file
            for file_path in dataset_path.glob('*.*'):
                file_stats = {
                    'size': file_path.stat().st_size,
                    'type': file_path.suffix[1:] if file_path.suffix else 'unknown',
                    'last_modified': datetime.fromtimestamp(
                        file_path.stat().st_mtime
                    ).isoformat()
                }

                # Additional analysis for CSV files
                if file_path.suffix.lower() == '.csv':
                    df = self.data_handler.read_csv(file_path)
                    file_stats.update({
                        'rows': len(df),
                        'columns': len(df.columns),
                        'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
                        'column_stats': self.data_handler.calculate_basic_stats(df)
                    })
                    total_rows += file_stats['rows']

                total_size += file_stats['size']
                stats[file_path.name] = file_stats

            # Add summary statistics
            stats['summary'] = {
                'total_files': len(stats),
                'total_size': total_size,
                'total_rows': total_rows,
                'downloaded': datetime.fromtimestamp(
                    dataset_path.stat().st_ctime
                ).isoformat()
            }

            return stats

        except Exception as e:
            logger.error(f"Error calculating dataset statistics: {str(e)}")
            raise

if __name__ == '__main__':
    # Example usage
    from src.api.kaggle_client import KaggleAPIClient
    from config.settings import get_kaggle_credentials

    credentials = get_kaggle_credentials()
    path_manager = PathManager()
    client = KaggleAPIClient(credentials)
    dataset_client = DatasetClient(client)
    manager = DatasetDownloadManager(dataset_client)


# https://www.kaggle.com/code/yash9439/health-insurance-claims-eda
    try:
        # Download dataset
        dataset_path = manager.download_dataset("health-insurance-claims-eda")
        print(f"Downloaded dataset to: {dataset_path}")

        # Process dataset
        process_config = {
            'handle_missing': True,
            'missing_strategy': {
                'numeric': 'mean',
                'categorical': 'mode'
            }
        }
        processed_path = manager.process_dataset(dataset_path, process_config)
        print(f"Processed dataset saved to: {processed_path}")

        # Get statistics
        stats = manager.get_dataset_stats(dataset_path)
        print(f"Dataset statistics: {stats}")

    except Exception as e:
        print(f"Error: {str(e)}")
