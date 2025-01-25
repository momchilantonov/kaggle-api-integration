from pathlib import Path
import yaml
import logging
from typing import Optional, Dict
import pandas as pd
from sklearn.ensemble import RandomForestClassifier #TODO
import joblib

from workflows.competition_workflows.submission_operations import CompetitionWorkflowManager
from workflows.dataset_workflows.download_operations import DatasetDownloadManager
from workflows.model_workflows.upload_operations import ModelUploadManager
from workflows.file_workflows.upload_operations import FileUploadManager
from src.handlers.data_handlers import DataHandler
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
        self._load_configs()

    def _load_configs(self):
        """Load necessary configurations"""
        with open('operational_configs/competition_configs/competition_params.yaml', 'r') as f:
            self.competition_config = yaml.safe_load(f)

        self.comp_settings = self.competition_config['active_competitions'][self.competition_name]

    @timer
    def run_pipeline(
        self,
        model_path: Optional[Path] = None,
        auto_submit: bool = False
    ) -> Dict:
        """
        Run complete competition pipeline

        Args:
            model_path: Optional path to existing model
            auto_submit: Whether to automatically submit predictions

        Returns:
            Pipeline results dictionary
        """
        try:
            # Step 1: Download competition data
            data_path = self._download_competition_data()
            logger.info(f"Downloaded competition data to {data_path}")

            # Step 2: Process data
            processed_data = self._process_data(data_path)
            logger.info("Data processing completed")

            # Step 3: Generate predictions
            predictions = self._generate_predictions(
                processed_data['test'],
                model_path,
                processed_data  # Pass complete processed data
            )
            logger.info("Generated predictions")

            # Step 4: Prepare submission
            submission_path = self._prepare_submission(predictions)
            logger.info(f"Prepared submission at {submission_path}")

            # Step 5: Submit predictions if requested
            submission_result = None
            if auto_submit:
                submission_result = self._submit_predictions(submission_path)
                logger.info(f"Submitted predictions: {submission_result}")

            return {
                'data_path': data_path,
                'processed_data': processed_data,
                'submission_path': submission_path,
                'submission_result': submission_result
            }

        except Exception as e:
            logger.error(f"Error in competition pipeline: {str(e)}")
            raise

    def _download_competition_data(self) -> Path:
        """Download competition data"""
        return self.competition_manager.download_competition_data(
            self.competition_name
        )

    def _process_data(self, data_path: Path) -> Dict:
        """Process competition data"""
        train_path = data_path / self.comp_settings['file_structure']['train']
        test_path = data_path / self.comp_settings['file_structure']['test']

        train_df = self.data_handler.read_csv(train_path)
        test_df = self.data_handler.read_csv(test_path)

        # Apply basic processing
        processing_config = {
            'handle_missing': True,
            'missing_strategy': {
                'numeric': 'mean',
                'categorical': 'mode'
            }
        }

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

    def _train_model(self, train_data: pd.DataFrame, target_column: str):
        """Train a new model"""
        X = train_data.drop(columns=[target_column])
        y = train_data[target_column]

        # Use simple RandomForest as default
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        model.fit(X, y)
        return model

    def _load_model(self, model_path: Path):
        """Load existing model"""
        return joblib.load(model_path)

    def _generate_predictions(
        self,
        test_data: pd.DataFrame,
        model_path: Optional[Path],
        processed_data: Dict
    ) -> pd.DataFrame:
        """Generate predictions using model"""
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

        predictions = model.predict(test_features)
        return pd.DataFrame({
            'id': test_data.index,
            'prediction': predictions
        })

    def _prepare_submission(self, predictions: pd.DataFrame) -> Path:
        """Prepare competition submission"""
        submission_path = Path(self.competition_config['data_paths']['submissions'])
        submission_path.mkdir(parents=True, exist_ok=True)

        file_path = submission_path / f"{self.competition_name}_submission.csv"
        self.data_handler.write_csv(predictions, file_path)

        return file_path

    def _submit_predictions(self, submission_path: Path) -> Dict:
        """Submit predictions to competition"""
        return self.competition_manager.submit_predictions(
            self.competition_name,
            submission_path,
            f"Submission from automated pipeline",
            wait_for_scoring=True
        )

    def get_pipeline_status(self) -> Dict:
        """Get current pipeline status"""
        competition_status = self.competition_manager.get_competition_status(
            self.competition_name
        )

        submission_history = self.competition_manager.get_submission_history(
            self.competition_name
        )

        return {
            'competition': competition_status,
            'submissions': submission_history,
            'deadline': self.comp_settings['deadline']
        }

def main():
    """Example usage of competition pipeline"""
    try:
        # Initialize pipeline for Titanic competition
        pipeline = CompetitionPipeline("titanic")

        # Run pipeline
        results = pipeline.run_pipeline(
            auto_submit=True
        )

        # Print results
        print("\nPipeline Results:")
        print(f"Data Path: {results['data_path']}")
        print(f"Submission Path: {results['submission_path']}")

        if results['submission_result']:
            print("\nSubmission Results:")
            print(f"Score: {results['submission_result'].get('score')}")
            print(f"Status: {results['submission_result'].get('status')}")

        # Get pipeline status
        status = pipeline.get_pipeline_status()
        print("\nPipeline Status:")
        print(f"Current Rank: {status['competition']['current_rank']}")
        print(f"Best Score: {status['competition']['best_score']}")
        print(f"Deadline: {status['deadline']}")

    except Exception as e:
        print(f"Error running pipeline: {str(e)}")

if __name__ == "__main__":
    main()
