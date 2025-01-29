from pathlib import Path
import yaml
import logging
from typing import Optional, Dict
import pandas as pd
from datetime import datetime
import json

from workflows.competition_workflows.submission_operations import CompetitionWorkflowManager
from workflows.dataset_workflows.download_operations import DatasetDownloadManager
from workflows.model_workflows.upload_operations import ModelUploadManager
from workflows.file_workflows.upload_operations import FileUploadManager
from src.handlers.data_handlers import DataHandler
from src.utils.path_manager import PathManager
from src.utils.error_handlers import handle_api_errors
from src.utils.helpers import timer

logger = logging.getLogger(__name__)

class CompetitionPipeline:
    """Complete pipeline for competition workflow"""

    def __init__(self, competition_name: str):
        """Initialize pipeline managers"""
        self.competition_name = competition_name
        self.competition_manager = CompetitionWorkflowManager()
        self.dataset_manager = DatasetDownloadManager()
        self.model_manager = ModelUploadManager()
        self.file_manager = FileUploadManager()
        self.data_handler = DataHandler()
        self.path_manager = PathManager()
        # Ensure required directories exist
        self.path_manager.ensure_directories()
        self._load_configs()

    def _load_configs(self):
        """Load necessary configurations"""
        try:
            comp_config_path = Path('operational_configs/competition_configs')

            # Load competition parameters
            with open(comp_config_path / 'competition_params.yaml', 'r') as f:
                self.competition_config = yaml.safe_load(f)

            # Load submission rules
            with open(comp_config_path / 'submission_rules.yaml', 'r') as f:
                self.submission_config = yaml.safe_load(f)

            self.comp_settings = self.competition_config['active_competitions'].get(
                self.competition_name,
                {}
            )
            if not self.comp_settings:
                raise ValueError(f"No configuration found for competition: {self.competition_name}")

            logger.info(f"Loaded configurations for competition: {self.competition_name}")

        except Exception as e:
            logger.error(f"Error loading competition configurations: {str(e)}")
            raise

    @timer
    @handle_api_errors
    def run_pipeline(
        self,
        model_path: Optional[Path] = None,
        auto_submit: bool = False,
        callback: Optional[callable] = None
    ) -> Dict:
        """Run complete competition pipeline"""
        try:
            if callback:
                callback(0, "Starting competition pipeline")

            # Step 1: Download competition data
            data_path = self._download_competition_data()
            logger.info(f"Downloaded competition data to {data_path}")
            if callback:
                callback(20, "Downloaded competition data")

            # Step 2: Process data
            processed_data = self._process_data(data_path)
            logger.info("Data processing completed")
            if callback:
                callback(40, "Processed competition data")

            # Step 3: Generate predictions
            predictions = self._generate_predictions(
                processed_data['test'],
                model_path,
                processed_data
            )
            logger.info("Generated predictions")
            if callback:
                callback(60, "Generated predictions")

            # Step 4: Prepare submission
            submission_path = self._prepare_submission(predictions)
            logger.info(f"Prepared submission at {submission_path}")
            if callback:
                callback(80, "Prepared submission")

            # Step 5: Submit predictions if requested
            submission_result = None
            if auto_submit:
                submission_result = self._submit_predictions(submission_path)
                logger.info(f"Submitted predictions: {submission_result}")
                if callback:
                    callback(100, "Submitted predictions")

            # Create pipeline result summary
            result = {
                'data_path': str(data_path),
                'processed_data_summary': self._create_data_summary(processed_data),
                'submission_path': str(submission_path),
                'submission_result': submission_result
            }

            # Log pipeline execution
            self._log_pipeline_execution(result)

            return result

        except Exception as e:
            logger.error(f"Error in competition pipeline: {str(e)}")
            raise

    @handle_api_errors
    def _download_competition_data(self) -> Path:
        """Download competition data"""
        try:
            comp_data_path = (
                self.path_manager.get_path('competitions', 'data') /
                self.competition_name
            )
            return self.competition_manager.download_competition_data(
                self.competition_name,
                custom_path=comp_data_path
            )
        except Exception as e:
            logger.error(f"Error downloading competition data: {str(e)}")
            raise

    def _process_data(self, data_path: Path) -> Dict:
        """Process competition data"""
        try:
            # Get file paths from competition settings
            train_path = data_path / self.comp_settings['file_structure']['train']
            test_path = data_path / self.comp_settings['file_structure']['test']

            if not train_path.exists() or not test_path.exists():
                raise FileNotFoundError("Required competition files not found")

            # Read data
            train_df = self.data_handler.read_csv(train_path)
            test_df = self.data_handler.read_csv(test_path)

            # Apply processing based on competition settings
            processing_config = self.comp_settings.get('processing_config', {
                'handle_missing': True,
                'missing_strategy': {
                    'numeric': 'mean',
                    'categorical': 'mode'
                }
            })

            train_processed = self.data_handler.handle_missing_values(
                train_df,
                processing_config['missing_strategy']
            )
            test_processed = self.data_handler.handle_missing_values(
                test_df,
                processing_config['missing_strategy']
            )

            return {
                'train': train_processed,
                'test': test_processed,
                'target_column': self.comp_settings['target_column']
            }

        except Exception as e:
            logger.error(f"Error processing competition data: {str(e)}")
            raise

    def _generate_predictions(
        self,
        test_data: pd.DataFrame,
        model_path: Optional[Path],
        processed_data: Dict
    ) -> pd.DataFrame:
        """Generate predictions using model"""
        try:
            if model_path:
                # Load and use existing model
                model = self._load_model(model_path)
            else:
                # Train new model
                model = self._train_model(
                    processed_data['train'],
                    processed_data['target_column']
                )

            # Remove target column if present in test data
            test_features = test_data.drop(
                columns=[processed_data['target_column']]
                if processed_data['target_column'] in test_data.columns
                else []
            )

            # Generate predictions
            predictions = model.predict(test_features)

            # Create submission DataFrame
            return pd.DataFrame({
                'id': test_data.index,
                processed_data['target_column']: predictions
            })

        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise

    def _train_model(self, train_data: pd.DataFrame, target_column: str):
        """Train a new model"""
        try:
            from sklearn.ensemble import RandomForestClassifier

            X = train_data.drop(columns=[target_column])
            y = train_data[target_column]

            # Get model parameters from competition settings
            model_params = self.comp_settings.get('model_params', {
                'n_estimators': 100,
                'random_state': 42,
                'n_jobs': -1
            })

            # Train model
            model = RandomForestClassifier(**model_params)
            model.fit(X, y)

            # Save model for future use
            self._save_model(model)

            return model

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def _save_model(self, model) -> Path:
        """Save trained model"""
        try:
            import joblib

            model_dir = (
                self.path_manager.get_path('competitions', 'models') /
                self.competition_name
            )
            model_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = model_dir / f"model_{timestamp}.joblib"

            joblib.dump(model, model_path)
            logger.info(f"Saved model to {model_path}")

            # Save model metadata
            metadata = {
                'timestamp': timestamp,
                'competition': self.competition_name,
                'model_type': type(model).__name__,
                'parameters': model.get_params()
            }

            with open(model_path.with_suffix('.meta.json'), 'w') as f:
                json.dump(metadata, f, indent=2)

            return model_path

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def _load_model(self, model_path: Path):
        """Load existing model"""
        try:
            import joblib

            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            model = joblib.load(model_path)
            logger.info(f"Loaded model from {model_path}")
            return model

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _prepare_submission(self, predictions: pd.DataFrame) -> Path:
        """Prepare competition submission"""
        try:
            # Get submission directory
            submission_dir = (
                self.path_manager.get_path('competitions', 'submissions') /
                self.competition_name
            )
            submission_dir.mkdir(parents=True, exist_ok=True)

            # Create submission file name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            submission_path = submission_dir / f"submission_{timestamp}.csv"

            # Validate submission format
            submission_rules = self.submission_config['submission_settings'].get(
                self.competition_name,
                self.submission_config['submission_settings']['default']
            )

            is_valid, errors = self.data_handler.validate_submission_format(
                predictions,
                required_columns=submission_rules['required_columns'],
                column_types=submission_rules.get('column_types', {})
            )

            if not is_valid:
                raise ValueError(f"Invalid submission format: {errors}")

            # Save submission file
            predictions.to_csv(submission_path, index=False)
            logger.info(f"Prepared submission file: {submission_path}")

            return submission_path

        except Exception as e:
            logger.error(f"Error preparing submission: {str(e)}")
            raise

    def _submit_predictions(self, submission_path: Path) -> Dict:
        """Submit predictions to competition"""
        try:
            result = self.competition_manager.submit_predictions(
                self.competition_name,
                submission_path,
                message=f"Pipeline submission at {datetime.now()}"
            )

            # Log submission
            self._log_submission(submission_path, result)

            return result

        except Exception as e:
            logger.error(f"Error submitting predictions: {str(e)}")
            raise

    def _create_data_summary(self, processed_data: Dict) -> Dict:
        """Create summary of processed data"""
        try:
            return {
                'train_shape': processed_data['train'].shape,
                'test_shape': processed_data['test'].shape,
                'target_column': processed_data['target_column'],
                'train_statistics': self.data_handler.calculate_basic_stats(
                    processed_data['train']
                ),
                'test_statistics': self.data_handler.calculate_basic_stats(
                    processed_data['test']
                )
            }
        except Exception as e:
            logger.error(f"Error creating data summary: {str(e)}")
            return {}

    def _log_pipeline_execution(self, result: Dict) -> None:
        """Log pipeline execution details"""
        try:
            log_dir = self.path_manager.get_path('logs')
            log_path = log_dir / f"{self.competition_name}_pipeline.log"

            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'competition': self.competition_name,
                'execution_result': result
            }

            with open(log_path, 'a') as f:
                f.write(f"{json.dumps(log_entry)}\n")

        except Exception as e:
            logger.error(f"Error logging pipeline execution: {str(e)}")

    def _log_submission(self, submission_path: Path, result: Dict) -> None:
        """Log submission details"""
        try:
            log_dir = self.path_manager.get_path('logs')
            log_path = log_dir / f"{self.competition_name}_submissions.log"

            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'submission_file': str(submission_path),
                'submission_id': result.get('id'),
                'status': result.get('status'),
                'score': result.get('score')
            }

            with open(log_path, 'a') as f:
                f.write(f"{json.dumps(log_entry)}\n")

        except Exception as e:
            logger.error(f"Error logging submission: {str(e)}")

if __name__ == '__main__':
    # Example usage
    try:
        pipeline = CompetitionPipeline("titanic")

        def progress_callback(progress, status):
            print(f"Progress: {progress}%, Status: {status}")

        result = pipeline.run_pipeline(
            auto_submit=True,
            callback=progress_callback
        )

        print("\nPipeline Results:")
        print(f"Data Path: {result['data_path']}")
        print("\nData Summary:")
        print(f"Train Shape: {result['processed_data_summary']['train_shape']}")
        print(f"Test Shape: {result['processed_data_summary']['test_shape']}")

        if result['submission_result']:
            print("\nSubmission Results:")
            print(f"Status: {result['submission_result'].get('status')}")
            print(f"Score: {result['submission_result'].get('score')}")

    except Exception as e:
        print(f"Error running pipeline: {str(e)}")
