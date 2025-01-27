from pathlib import Path
import yaml
import logging
from typing import Optional, Dict, List
import pandas as pd

from workflows.dataset_workflows.download_operations import DatasetDownloadManager
from workflows.dataset_workflows.upload_operations import DatasetUploadManager
from workflows.file_workflows.upload_operations import FileUploadManager
from src.handlers.data_handlers import DataHandler
from src.utils.helpers import timer

logger = logging.getLogger(__name__)

class DatasetPipeline:
    """Complete pipeline for dataset operations"""

    def __init__(self):
        self.dataset_download_manager = DatasetDownloadManager()
        self.dataset_upload_manager = DatasetUploadManager()
        self.file_manager = FileUploadManager()
        self.data_handler = DataHandler()
        self._load_configs()

    def _load_configs(self):
        with open('operational_configs/dataset_configs/datasets.yaml', 'r') as f:
            self.dataset_config = yaml.safe_load(f)

    @timer
    def process_dataset(
        self,
        dataset_name: str,
        process_config: Dict,
        upload: bool = False
    ) -> Dict:
        """Process a dataset with optional upload"""
        try:
            # Download dataset
            dataset_path = self._download_dataset(dataset_name)

            # Process files
            processed_path = self._process_files(dataset_path, process_config)

            # Upload if requested
            upload_result = None
            if upload:
                upload_result = self._upload_processed_dataset(
                    processed_path,
                    f"{dataset_name}_processed"
                )

            return {
                'original_path': dataset_path,
                'processed_path': processed_path,
                'upload_result': upload_result,
                'statistics': self._calculate_statistics(processed_path)
            }

        except Exception as e:
            logger.error(f"Error in dataset pipeline: {str(e)}")
            raise

    def _download_dataset(self, dataset_name: str) -> Path:
        """Download dataset files"""
        dataset_info = self.dataset_config['frequently_used_datasets'][dataset_name]
        return self.dataset_download_manager.download_dataset(
            dataset_info['owner'],
            dataset_info['dataset'],
            Path("data/datasets/raw") / dataset_name
        )

    def _process_files(self, dataset_path: Path, process_config: Dict) -> Path:
        """Process dataset files according to configuration"""
        processed_path = Path("data/datasets/processed") / dataset_path.name
        processed_path.mkdir(parents=True, exist_ok=True)

        for file_path in dataset_path.glob('*.csv'):
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
            logger.info(f"Processed {file_path.name}")

        return processed_path

    def _upload_processed_dataset(
        self,
        dataset_path: Path,
        dataset_name: str
    ) -> Dict:
        """Upload processed dataset"""
        # Prepare dataset for upload
        prepared_dir = self.dataset_upload_manager.prepare_dataset_folder(
            list(dataset_path.glob('*.csv')),
            dataset_name
        )

        # Create metadata
        metadata = self.dataset_upload_manager.create_metadata(
            title=dataset_name,
            description=f"Processed version of {dataset_path.name}",
            licenses=[{"name": "CC0-1.0"}],
            keywords=["processed", "cleaned"]
        )

        # Upload dataset
        return self.dataset_upload_manager.upload_dataset(
            prepared_dir,
            metadata
        )

    def _calculate_statistics(self, dataset_path: Path) -> Dict:
        """Calculate statistics for processed dataset"""
        stats = {}
        for file_path in dataset_path.glob('*.csv'):
            df = self.data_handler.read_csv(file_path)
            stats[file_path.name] = self.data_handler.calculate_basic_stats(df)
        return stats

    @timer
    def batch_process_datasets(
        self,
        dataset_names: List[str],
        process_config: Dict
    ) -> Dict[str, Dict]:
        """Process multiple datasets in batch"""
        results = {}
        for dataset_name in dataset_names:
            try:
                results[dataset_name] = self.process_dataset(
                    dataset_name,
                    process_config
                )
            except Exception as e:
                logger.error(f"Error processing {dataset_name}: {str(e)}")
                results[dataset_name] = {'error': str(e)}
        return results

def main():
    """Example usage of dataset pipeline"""
    try:
        pipeline = DatasetPipeline()

        # Define processing configuration
        process_config = {
            'handle_missing': True,
            'missing_strategy': {
                'numeric': 'mean',
                'categorical': 'mode'
            },
            'convert_types': True,
            'type_mapping': {
                'id': int,
                'value': float
            }
        }

        # Process single dataset
        result = pipeline.process_dataset(
            "titanic",
            process_config,
            upload=True
        )

        print("\nProcessing Results:")
        print(f"Original Path: {result['original_path']}")
        print(f"Processed Path: {result['processed_path']}")

        if result['upload_result']:
            print(f"\nUpload Result: {result['upload_result']}")

        print("\nDataset Statistics:")
        for file_name, stats in result['statistics'].items():
            print(f"\n{file_name}:")
            for col, col_stats in stats.items():
                print(f"  {col}: {col_stats}")

    except Exception as e:
        print(f"Error running pipeline: {str(e)}")

if __name__ == "__main__":
    main()
