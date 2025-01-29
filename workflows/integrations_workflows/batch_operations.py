from pathlib import Path
import yaml
import logging
import joblib
from typing import Dict, List, Optional, Union, Callable
import concurrent.futures
from datetime import datetime
import json
from sklearn.ensemble import RandomForestClassifier

from workflows.dataset_workflows.download_operations import DatasetDownloadManager
from workflows.model_workflows.upload_operations import ModelUploadManager
from workflows.competition_workflows.submission_operations import CompetitionWorkflowManager
from workflows.file_workflows.upload_operations import FileUploadManager
from src.handlers.data_handlers import DataHandler
from src.utils.path_manager import PathManager
from src.utils.error_handlers import handle_api_errors, retry_with_backoff
from src.utils.helpers import timer

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

class BatchOperationsManager:
    """Manages batch operations across different workflows"""

    def __init__(self):
        # Initialize workflow managers
        self.dataset_manager = DatasetDownloadManager()
        self.model_manager = ModelUploadManager()
        self.competition_manager = CompetitionWorkflowManager()
        self.file_manager = FileUploadManager()

        # Initialize handlers
        self.data_handler = DataHandler()
        self.path_manager = PathManager()

        # Ensure required directories exist
        self.path_manager.ensure_directories()

        # Load configurations
        self._load_configs()

    def _load_configs(self):
        """Load operational configurations"""
        try:
            configs_path = Path('operational_configs')
            self.configs = {}
            for config_file in configs_path.glob('**/*.yaml'):
                with open(config_file) as f:
                    category = config_file.parent.name
                    name = config_file.stem
                    if category not in self.configs:
                        self.configs[category] = {}
                    self.configs[category][name] = yaml.safe_load(f)
            logger.info("Successfully loaded operational configurations")
        except Exception as e:
            logger.error(f"Error loading configurations: {str(e)}")
            raise

    @timer
    @handle_api_errors
    def batch_download_datasets(
        self,
        dataset_list: List[str],
        process_config: Optional[Dict] = None,
        max_workers: int = 4,
        callback: Optional[Callable] = None
    ) -> Dict:
        """Download and optionally process multiple datasets"""
        try:
            results = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Create download futures
                future_to_dataset = {
                    executor.submit(
                        self._download_and_process_dataset,
                        dataset,
                        process_config,
                        callback
                    ): dataset
                    for dataset in dataset_list
                }

                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_dataset):
                    dataset = future_to_dataset[future]
                    try:
                        results[dataset] = future.result()
                        logger.info(f"Successfully processed dataset: {dataset}")
                    except Exception as e:
                        results[dataset] = {'error': str(e)}
                        logger.error(f"Error processing dataset {dataset}: {str(e)}")

            # Log batch operation results
            self._log_batch_operation('dataset_downloads', results)
            return results

        except Exception as e:
            logger.error(f"Error in batch dataset downloads: {str(e)}")
            raise

    def _download_and_process_dataset(
        self,
        dataset_name: str,
        process_config: Optional[Dict],
        callback: Optional[Callable]
    ) -> Dict:
        """Download and process a single dataset"""
        try:
            # Update progress
            if callback:
                callback(0, f"Starting download of {dataset_name}")

            # Download dataset
            dataset_path = self.dataset_manager.download_dataset(dataset_name)

            if callback:
                callback(50, f"Processing {dataset_name}")

            # Process if config provided
            if process_config:
                processed_path = self.dataset_manager.process_dataset(
                    dataset_path,
                    process_config
                )
                result = {
                    'status': 'success',
                    'raw_path': str(dataset_path),
                    'processed_path': str(processed_path)
                }
            else:
                result = {
                    'status': 'success',
                    'path': str(dataset_path)
                }

            if callback:
                callback(100, f"Completed {dataset_name}")

            return result

        except Exception as e:
            logger.error(f"Error processing dataset {dataset_name}: {str(e)}")
            raise

    @timer
    @handle_api_errors
    def batch_train_models(
        self,
        training_configs: List[Dict],
        max_workers: int = 2,
        callback: Optional[Callable] = None
    ) -> Dict:
        """Train multiple models in batch"""
        try:
            results = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Create training futures
                future_to_model = {
                    executor.submit(
                        self._train_single_model,
                        config,
                        callback
                    ): config['name']
                    for config in training_configs
                }

                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_model):
                    model_name = future_to_model[future]
                    try:
                        results[model_name] = future.result()
                        logger.info(f"Successfully trained model: {model_name}")
                    except Exception as e:
                        results[model_name] = {'error': str(e)}
                        logger.error(f"Error training model {model_name}: {str(e)}")

            # Log batch operation results
            self._log_batch_operation('model_training', results)
            return results

        except Exception as e:
            logger.error(f"Error in batch model training: {str(e)}")
            raise

    def _train_single_model(
        self,
        config: Dict,
        callback: Optional[Callable]
    ) -> Dict:
        """Train a single model"""
        try:
            if callback:
                callback(0, f"Starting training of {config['name']}")

            # Load and prepare data
            data = self.data_handler.read_csv(config['data_path'])
            processed_data = self.data_handler.handle_missing_values(
                data,
                config['missing_strategy']
            )

            if callback:
                callback(30, f"Data prepared for {config['name']}")

            # Train model
            model = config['model_class'](**config['model_params'])
            model.fit(
                processed_data.drop(columns=[config['target_column']]),
                processed_data[config['target_column']]
            )

            if callback:
                callback(70, f"Model trained for {config['name']}")

            # Save model
            save_path = (
                self.path_manager.get_path('models', 'custom') /
                f"{config['name']}_model.joblib"
            )
            joblib.dump(model, save_path)

            if callback:
                callback(100, f"Completed {config['name']}")

            return {
                'status': 'success',
                'model_path': str(save_path)
            }

        except Exception as e:
            logger.error(f"Error training model {config['name']}: {str(e)}")
            raise

    @timer
    @handle_api_errors
    def batch_submit_to_competitions(
        self,
        submission_configs: List[Dict],
        max_workers: int = 2,
        callback: Optional[Callable] = None
    ) -> Dict:
        """Submit multiple predictions to competitions"""
        try:
            results = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Create submission futures
                future_to_submission = {
                    executor.submit(
                        self._submit_to_competition,
                        config,
                        callback
                    ): config['competition']
                    for config in submission_configs
                }

                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_submission):
                    competition = future_to_submission[future]
                    try:
                        results[competition] = future.result()
                        logger.info(f"Successfully submitted to competition: {competition}")
                    except Exception as e:
                        results[competition] = {'error': str(e)}
                        logger.error(f"Error submitting to {competition}: {str(e)}")

            # Log batch operation results
            self._log_batch_operation('competition_submissions', results)
            return results

        except Exception as e:
            logger.error(f"Error in batch competition submissions: {str(e)}")
            raise

    def _submit_to_competition(
        self,
        config: Dict,
        callback: Optional[Callable]
    ) -> Dict:
        """Submit to a single competition"""
        try:
            if callback:
                callback(0, f"Starting submission to {config['competition']}")

            # Load and validate predictions
            predictions = self.data_handler.read_csv(config['predictions_path'])

            if callback:
                callback(30, f"Validating submission for {config['competition']}")

            # Prepare submission
            submission_path = self.competition_manager.prepare_submission(
                predictions,
                config['competition']
            )

            if callback:
                callback(60, f"Submitting to {config['competition']}")

            # Submit predictions
            result = self.competition_manager.submit_predictions(
                config['competition'],
                submission_path,
                config.get('message', f"Batch submission {datetime.now()}")
            )

            if callback:
                callback(100, f"Completed submission to {config['competition']}")

            return {
                'status': 'success',
                'submission_id': result.get('id'),
                'score': result.get('score')
            }

        except Exception as e:
            logger.error(f"Error submitting to competition {config['competition']}: {str(e)}")
            raise

    @timer
    @handle_api_errors
    def batch_file_operations(
        self,
        operations: List[Dict],
        max_workers: int = 4,
        callback: Optional[Callable] = None
    ) -> Dict:
        """Process multiple file operations in batch"""
        try:
            results = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Create operation futures
                future_to_operation = {
                    executor.submit(
                        self._process_file_operation,
                        op,
                        callback
                    ): op['file']
                    for op in operations
                }

                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_operation):
                    file_name = future_to_operation[future]
                    try:
                        results[file_name] = future.result()
                        logger.info(f"Successfully processed file operation: {file_name}")
                    except Exception as e:
                        results[file_name] = {'error': str(e)}
                        logger.error(f"Error processing file {file_name}: {str(e)}")

            # Log batch operation results
            self._log_batch_operation('file_operations', results)
            return results

        except Exception as e:
            logger.error(f"Error in batch file operations: {str(e)}")
            raise

    def _process_file_operation(
        self,
        operation: Dict,
        callback: Optional[Callable]
    ) -> Dict:
        """Process a single file operation"""
        try:
            op_type = operation['type']
            if callback:
                callback(0, f"Starting {op_type} operation for {operation['file']}")

            if op_type == 'upload':
                result = self.file_manager.upload_file(
                    operation['file_path'],
                    operation['dataset_owner'],
                    operation['dataset_name']
                )
            elif op_type == 'download':
                result = self.file_manager.get_file(
                    operation['dataset_owner'],
                    operation['dataset_name'],
                    operation['file']
                )
            else:
                raise ValueError(f"Unknown operation type: {op_type}")

            if callback:
                callback(100, f"Completed {op_type} operation for {operation['file']}")

            return {
                'status': 'success',
                'operation_type': op_type,
                'result': result
            }

        except Exception as e:
            logger.error(f"Error processing file operation: {str(e)}")
            raise

    def _log_batch_operation(self, operation_type: str, results: Dict) -> None:
        """Log batch operation results"""
        try:
            log_dir = self.path_manager.get_path('logs')
            log_path = log_dir / f"batch_{operation_type}.log"

            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'operation_type': operation_type,
                'total_operations': len(results),
                'successful': len([r for r in results.values() if 'error' not in r]),
                'failed': len([r for r in results.values() if 'error' in r]),
                'results': results
            }

            with open(log_path, 'a') as f:
                f.write(f"{json.dumps(log_entry)}\n")

        except Exception as e:
            logger.error(f"Error logging batch operation: {str(e)}")

if __name__ == '__main__':
    # Example usage
    manager = BatchOperationsManager()

    try:
        # Example: Batch dataset downloads
        datasets = ["titanic", "house-prices"]
        download_results = manager.batch_download_datasets(datasets)
        print("\nDataset Download Results:")
        for dataset, result in download_results.items():
            print(f"{dataset}: {result['status']}")

        # Example: Batch model training
        training_configs = [
            {
                'name': 'titanic_rf',
                'data_path': 'data/datasets/titanic/train.csv',
                'model_class': RandomForestClassifier,
                'model_params': {'n_estimators': 100},
                'target_column': 'Survived',
                'missing_strategy': {'numeric': 'mean', 'categorical': 'mode'}
            }
        ]
        training_results = manager.batch_train_models(training_configs)
        print("\nModel Training Results:")
        for model, result in training_results.items():
            print(f"{model}: {result['status']}")

        # Example: Batch competition submissions
        submission_configs = [
            {
                'competition': 'titanic',
                'predictions_path': 'data/predictions/titanic_predictions.csv',
                'message': 'Batch submission test'
            }
        ]
        submission_results = manager.batch_submit_to_competitions(submission_configs)
        print("\nSubmission Results:")
        for comp, result in submission_results.items():
            print(f"{comp}: {result['status']}")

    except Exception as e:
        print(f"Error in batch operations: {str(e)}")
