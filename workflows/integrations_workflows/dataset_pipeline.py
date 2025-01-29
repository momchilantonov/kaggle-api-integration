from pathlib import Path
import yaml
import logging
from typing import Optional, Dict, List
import pandas as pd
from datetime import datetime
import json

from workflows.dataset_workflows.download_operations import DatasetDownloadManager
from workflows.dataset_workflows.upload_operations import DatasetUploadManager
from workflows.file_workflows.upload_operations import FileUploadManager
from src.handlers.data_handlers import DataHandler
from src.utils.path_manager import PathManager
from src.utils.error_handlers import handle_api_errors
from src.utils.helpers import timer

logger = logging.getLogger(__name__)

class DatasetPipeline:
    """Complete pipeline for dataset operations"""

    def __init__(self):
        self.dataset_download_manager = DatasetDownloadManager()
        self.dataset_upload_manager = DatasetUploadManager()
        self.file_manager = FileUploadManager()
        self.data_handler = DataHandler()
        self.path_manager = PathManager()
        # Ensure required directories exist
        self.path_manager.ensure_directories()
        self._load_configs()

    def _load_configs(self):
        """Load operational configurations"""
        try:
            dataset_config_path = Path('operational_configs/dataset_configs')

            # Load dataset configurations
            with open(dataset_config_path / 'datasets.yaml', 'r') as f:
                self.dataset_config = yaml.safe_load(f)

            # Load download settings
            with open(dataset_config_path / 'download_settings.yaml', 'r') as f:
                self.download_config = yaml.safe_load(f)

            logger.info("Successfully loaded dataset configurations")

        except Exception as e:
            logger.error(f"Error loading dataset configurations: {str(e)}")
            raise

    @timer
    @handle_api_errors
    def process_dataset(
        self,
        dataset_name: str,
        process_config: Dict,
        upload: bool = False,
        callback: Optional[callable] = None
    ) -> Dict:
        """Process a dataset with optional upload"""
        try:
            if callback:
                callback(0, "Starting dataset processing")

            # Download dataset
            dataset_path = self._download_dataset(dataset_name)
            if callback:
                callback(30, "Dataset downloaded")

            # Process files
            processed_path = self._process_files(dataset_path, process_config)
            if callback:
                callback(60, "Files processed")

            # Upload if requested
            upload_result = None
            if upload:
                upload_result = self._upload_processed_dataset(
                    processed_path,
                    f"{dataset_name}_processed"
                )
                if callback:
                    callback(90, "Dataset uploaded")

            # Calculate statistics
            statistics = self._calculate_statistics(processed_path)
            if callback:
                callback(100, "Processing completed")

            # Create result summary
            result = {
                'original_path': str(dataset_path),
                'processed_path': str(processed_path),
                'upload_result': upload_result,
                'statistics': statistics
            }

            # Log pipeline execution
            self._log_pipeline_execution(dataset_name, result)

            return result

        except Exception as e:
            logger.error(f"Error in dataset pipeline: {str(e)}")
            raise

    def _download_dataset(self, dataset_name: str) -> Path:
        """Download dataset files"""
        try:
            # Get dataset info from config
            dataset_info = self.dataset_config['frequently_used_datasets'].get(dataset_name)
            if not dataset_info:
                raise ValueError(f"Dataset {dataset_name} not found in configurations")

            # Download to appropriate directory
            dataset_path = self.dataset_download_manager.download_dataset(
                dataset_name=dataset_name,
                custom_path=self.path_manager.get_path('datasets', 'raw') / dataset_name
            )

            logger.info(f"Downloaded dataset to {dataset_path}")
            return dataset_path

        except Exception as e:
            logger.error(f"Error downloading dataset: {str(e)}")
            raise

    def _process_files(self, dataset_path: Path, process_config: Dict) -> Path:
        """Process dataset files according to configuration"""
        try:
            # Create processed data directory
            processed_path = self.path_manager.get_path('datasets', 'processed') / dataset_path.name
            processed_path.mkdir(parents=True, exist_ok=True)

            # Process each CSV file
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

                # Apply custom transformations if specified
                if process_config.get('transformations'):
                    df = self._apply_transformations(df, process_config['transformations'])

                # Save processed file
                output_path = processed_path / f"processed_{file_path.name}"
                self.data_handler.write_csv(df, output_path)
                logger.info(f"Processed {file_path.name}")

                # Create processing metadata
                self._create_processing_metadata(
                    output_path,
                    file_path,
                    process_config
                )

            return processed_path

        except Exception as e:
            logger.error(f"Error processing files: {str(e)}")
            raise

    def _apply_transformations(
        self,
        df: pd.DataFrame,
        transformations: List[Dict]
    ) -> pd.DataFrame:
        """Apply custom transformations to DataFrame"""
        try:
            for transform in transformations:
                transform_type = transform['type']
                columns = transform.get('columns', [])
                params = transform.get('params', {})

                if transform_type == 'normalize':
                    for col in columns:
                        if col in df.columns:
                            df[col] = (df[col] - df[col].mean()) / df[col].std()

                elif transform_type == 'one_hot_encode':
                    for col in columns:
                        if col in df.columns:
                            one_hot = pd.get_dummies(df[col], prefix=col)
                            df = pd.concat([df, one_hot], axis=1)
                            df.drop(columns=[col], inplace=True)

                elif transform_type == 'bin':
                    for col in columns:
                        if col in df.columns:
                            df[f"{col}_binned"] = pd.qcut(
                                df[col],
                                q=params.get('bins', 4),
                                labels=params.get('labels')
                            )

                else:
                    logger.warning(f"Unknown transformation type: {transform_type}")

            return df

        except Exception as e:
            logger.error(f"Error applying transformations: {str(e)}")
            raise

    def _create_processing_metadata(
        self,
        output_path: Path,
        input_path: Path,
        process_config: Dict
    ) -> None:
        """Create metadata about file processing"""
        try:
            metadata = {
                'original_file': str(input_path),
                'processed_file': str(output_path),
                'processing_date': datetime.now().isoformat(),
                'process_config': process_config,
                'file_stats': {
                    'original_size': input_path.stat().st_size,
                    'processed_size': output_path.stat().st_size
                }
            }

            metadata_path = output_path.with_suffix('.meta.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            logger.error(f"Error creating processing metadata: {str(e)}")
            raise

    @handle_api_errors
    def _upload_processed_dataset(
        self,
        dataset_path: Path,
        dataset_name: str
    ) -> Dict:
        """Upload processed dataset"""
        try:
            # Prepare dataset folder for upload
            prepared_dir = self.dataset_upload_manager.prepare_dataset_folder(
                list(dataset_path.glob('**/*.csv')),
                dataset_name
            )

            # Create metadata
            metadata = self.dataset_upload_manager.create_metadata(
                title=dataset_name,
                slug=dataset_name.lower().replace('_', '-'),
                description=self._generate_dataset_description(dataset_path),
                licenses=[{"name": "CC0-1.0"}],
                keywords=["processed", "cleaned", "transformed"]
            )

            # Upload dataset
            result = self.dataset_upload_manager.upload_dataset(
                prepared_dir,
                metadata
            )

            # Log upload
            self._log_upload(dataset_name, result)

            return result

        except Exception as e:
            logger.error(f"Error uploading processed dataset: {str(e)}")
            raise

    def _generate_dataset_description(self, dataset_path: Path) -> str:
        """Generate description for processed dataset"""
        try:
            stats = self._calculate_statistics(dataset_path)
            files = list(dataset_path.glob('**/*.csv'))

            description = (
                f"Processed dataset containing {len(files)} files.\n\n"
                f"## Files\n"
            )

            for file_path in files:
                meta_path = file_path.with_suffix('.meta.json')
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                    description += (
                        f"\n### {file_path.name}\n"
                        f"- Original file: {meta['original_file']}\n"
                        f"- Processing date: {meta['processing_date']}\n"
                    )

            description += "\n## Statistics\n"
            for file_name, file_stats in stats.items():
                description += f"\n### {file_name}\n"
                for stat_name, stat_value in file_stats.items():
                    description += f"- {stat_name}: {stat_value}\n"

            return description

        except Exception as e:
            logger.error(f"Error generating dataset description: {str(e)}")
            return "Processed dataset"

    def _calculate_statistics(self, dataset_path: Path) -> Dict:
        """Calculate comprehensive dataset statistics"""
        try:
            stats = {}
            for file_path in dataset_path.glob('**/*.csv'):
                df = self.data_handler.read_csv(file_path)
                stats[file_path.name] = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                    'missing_values': df.isnull().sum().sum(),
                    'column_types': df.dtypes.to_dict(),
                    'column_stats': self.data_handler.calculate_basic_stats(df)
                }
            return stats

        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            raise

    def _log_pipeline_execution(self, dataset_name: str, result: Dict) -> None:
        """Log pipeline execution details"""
        try:
            log_dir = self.path_manager.get_path('logs')
            log_path = log_dir / 'dataset_pipeline.log'

            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'dataset_name': dataset_name,
                'execution_result': result
            }

            with open(log_path, 'a') as f:
                f.write(f"{json.dumps(log_entry)}\n")

        except Exception as e:
            logger.error(f"Error logging pipeline execution: {str(e)}")

    def _log_upload(self, dataset_name: str, result: Dict) -> None:
        """Log dataset upload details"""
        try:
            log_dir = self.path_manager.get_path('logs')
            log_path = log_dir / 'dataset_uploads.log'

            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'dataset_name': dataset_name,
                'upload_result': result
            }

            with open(log_path, 'a') as f:
                f.write(f"{json.dumps(log_entry)}\n")

        except Exception as e:
            logger.error(f"Error logging upload: {str(e)}")

if __name__ == '__main__':
    # Example usage
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
                'id': 'int',
                'value': 'float'
            },
            'transformations': [
                {
                    'type': 'normalize',
                    'columns': ['age', 'salary']
                },
                {
                    'type': 'one_hot_encode',
                    'columns': ['category', 'department']
                }
            ]
        }

        def progress_callback(progress, status):
            print(f"Progress: {progress}%, Status: {status}")

        # Process dataset
        result = pipeline.process_dataset(
            "example_dataset",
            process_config,
            upload=True,
            callback=progress_callback
        )

        print("\nProcessing Results:")
        print(f"Original Path: {result['original_path']}")
        print(f"Processed Path: {result['processed_path']}")

        if result['upload_result']:
            print(f"\nUpload Result: {result['upload_result']}")

        print("\nDataset Statistics:")
        for file_name, stats in result['statistics'].items():
            print(f"\n{file_name}:")
            for stat_name, stat_value in stats.items():
                print(f"  {stat_name}: {stat_value}")

    except Exception as e:
        print(f"Error running pipeline: {str(e)}")
