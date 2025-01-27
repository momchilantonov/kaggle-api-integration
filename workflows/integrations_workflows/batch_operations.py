from pathlib import Path
import yaml
import logging
import joblib
from typing import Dict, List
import concurrent.futures
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

from workflows.dataset_workflows.download_operations import DatasetDownloadManager
from workflows.model_workflows.upload_operations import ModelUploadManager
from workflows.competition_workflows.submission_operations import CompetitionWorkflowManager
from workflows.file_workflows.upload_operations import FileUploadManager
from src.handlers.data_handlers import DataHandler
from src.utils.helpers import timer

logger = logging.getLogger(__name__)

class BatchOperationsManager:
    def __init__(self):
        self.dataset_manager = DatasetDownloadManager()
        self.model_manager = ModelUploadManager()
        self.competition_manager = CompetitionWorkflowManager()
        self.file_manager = FileUploadManager()
        self.data_handler = DataHandler()
        self._load_configs()

    def _load_configs(self):
        configs_path = Path('operational_configs')
        self.configs = {}
        for config_file in configs_path.glob('**/*.yaml'):
            with open(config_file) as f:
                category = config_file.parent.name
                name = config_file.stem
                if category not in self.configs:
                    self.configs[category] = {}
                self.configs[category][name] = yaml.safe_load(f)

    @timer
    def batch_download_datasets(self, dataset_list: List[str]) -> Dict:
        results = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_dataset = {
                executor.submit(self._download_single_dataset, dataset): dataset
                for dataset in dataset_list
            }

            for future in concurrent.futures.as_completed(future_to_dataset):
                dataset = future_to_dataset[future]
                try:
                    results[dataset] = future.result()
                except Exception as e:
                    results[dataset] = {'error': str(e)}

        return results

    def _download_single_dataset(self, dataset_name: str) -> Dict:
        try:
            dataset_path = self.dataset_manager.download_dataset(dataset_name)
            return {'status': 'success', 'path': str(dataset_path)}
        except Exception as e:
            logger.error(f"Error downloading {dataset_name}: {str(e)}")
            raise

    @timer
    def batch_train_models(
        self,
        training_configs: List[Dict]
    ) -> Dict:
        results = {}
        for config in training_configs:
            try:
                results[config['name']] = self._train_single_model(config)
            except Exception as e:
                results[config['name']] = {'error': str(e)}
        return results

    def _train_single_model(self, config: Dict) -> Dict:
        try:
            # Load and prepare data
            data = self.data_handler.read_csv(config['data_path'])
            processed_data = self.data_handler.handle_missing_values(
                data,
                config['missing_strategy']
            )

            # Train model
            model = config['model_class'](**config['model_params'])
            model.fit(
                processed_data.drop(columns=[config['target_column']]),
                processed_data[config['target_column']]
            )

            # Save model
            save_path = Path(config['save_path'])
            save_path.mkdir(parents=True, exist_ok=True)
            model_path = save_path / f"{config['name']}_model.joblib"
            joblib.dump(model, model_path)

            return {
                'status': 'success',
                'model_path': str(model_path)
            }
        except Exception as e:
            logger.error(f"Error training model {config['name']}: {str(e)}")
            raise

    @timer
    def batch_submit_to_competitions(
        self,
        submission_configs: List[Dict]
    ) -> Dict:
        results = {}
        for config in submission_configs:
            try:
                results[config['competition']] = self._submit_to_competition(config)
            except Exception as e:
                results[config['competition']] = {'error': str(e)}
        return results

    def _submit_to_competition(self, config: Dict) -> Dict:
        try:
            predictions = self.data_handler.read_csv(config['predictions_path'])
            submission_path = self.competition_manager.prepare_submission(
                predictions,
                config['competition']
            )
            result = self.competition_manager.submit_predictions(
                config['competition'],
                submission_path,
                config.get('message', f"Batch submission {datetime.now()}")
            )
            return {'status': 'success', 'result': result}
        except Exception as e:
            logger.error(f"Error submitting to {config['competition']}: {str(e)}")
            raise

    @timer
    def batch_file_operations(
        self,
        operations: List[Dict]
    ) -> Dict:
        results = {}
        for op in operations:
            try:
                if op['type'] == 'upload':
                    results[op['file']] = self._upload_file(op)
                elif op['type'] == 'download':
                    results[op['file']] = self._download_file(op)
            except Exception as e:
                results[op['file']] = {'error': str(e)}
        return results

    def _upload_file(self, config: Dict) -> Dict:
        try:
            result = self.file_manager.upload_file(
                config['file_path'],
                config['dataset_owner'],
                config['dataset_name']
            )
            return {'status': 'success', 'result': result}
        except Exception as e:
            logger.error(f"Error uploading {config['file_path']}: {str(e)}")
            raise

    def _download_file(self, config: Dict) -> Dict:
        try:
            path = self.file_manager.get_file(
                config['dataset_owner'],
                config['dataset_name'],
                config['file_name'],
                config.get('download_path')
            )
            return {'status': 'success', 'path': str(path)}
        except Exception as e:
            logger.error(f"Error downloading {config['file_name']}: {str(e)}")
            raise

def main():
    try:
        batch_manager = BatchOperationsManager()

        # Example: Batch dataset downloads
        datasets = ["titanic", "house-prices"]
        download_results = batch_manager.batch_download_datasets(datasets)
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
                'missing_strategy': {'numeric': 'mean', 'categorical': 'mode'},
                'save_path': 'data/models/batch'
            }
        ]
        training_results = batch_manager.batch_train_models(training_configs)
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
        submission_results = batch_manager.batch_submit_to_competitions(submission_configs)
        print("\nSubmission Results:")
        for comp, result in submission_results.items():
            print(f"{comp}: {result['status']}")

    except Exception as e:
        print(f"Error in batch operations: {str(e)}")

if __name__ == "__main__":
    main()
