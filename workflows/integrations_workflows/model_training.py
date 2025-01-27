from pathlib import Path
import yaml
import logging
from typing import Optional, Dict, List
import pandas as pd
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # type: ignore
import joblib

from workflows.dataset_workflows.download_operations import DatasetDownloadManager
from workflows.model_workflows.upload_operations import ModelUploadManager
from src.handlers.data_handlers import DataHandler
from src.utils.helpers import timer

logger = logging.getLogger(__name__)

class ModelTrainingPipeline:
    def __init__(self):
        self.dataset_manager = DatasetDownloadManager()
        self.model_manager = ModelUploadManager()
        self.data_handler = DataHandler()
        self._load_configs()

    def _load_configs(self):
        with open('operational_configs/model_configs/training_config.yaml', 'r') as f:
            self.training_config = yaml.safe_load(f)
        with open('operational_configs/model_configs/model_params.yaml', 'r') as f:
            self.model_config = yaml.safe_load(f)

    @timer
    def train_model(
        self,
        dataset_name: str,
        model_type: str,
        target_column: str,
        custom_params: Optional[Dict] = None
    ) -> Dict:
        try:
            # Prepare data
            train_data, val_data = self._prepare_data(
                dataset_name,
                target_column
            )

            # Train model
            model, metrics = self._train_and_evaluate(
                train_data,
                val_data,
                target_column,
                model_type,
                custom_params
            )

            # Save model
            model_path = self._save_model(model, model_type)

            # Upload if configured
            upload_result = None
            if self.model_config.get('auto_upload', False):
                upload_result = self._upload_model(
                    model_path,
                    metrics,
                    model_type
                )

            return {
                'model_path': model_path,
                'metrics': metrics,
                'upload_result': upload_result
            }

        except Exception as e:
            logger.error(f"Error in model training pipeline: {str(e)}")
            raise

    def _prepare_data(
        self,
        dataset_name: str,
        target_column: str
    ) -> tuple:
        # Download and load data
        dataset_path = self.dataset_manager.download_dataset(dataset_name)
        data = self.data_handler.read_csv(dataset_path / "train.csv")

        # Handle missing values
        data = self.data_handler.handle_missing_values(
            data,
            self.training_config['data_handling']['preprocessing']['handle_missing']
        )

        # Split data
        train_size = self.training_config['training_settings']['default']['validation_split']
        return train_test_split(
            data,
            test_size=1-train_size,
            random_state=42
        )

    def _train_and_evaluate(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        target_column: str,
        model_type: str,
        custom_params: Optional[Dict]
    ) -> tuple:
        # Prepare features and target
        X_train = train_data.drop(columns=[target_column])
        y_train = train_data[target_column]
        X_val = val_data.drop(columns=[target_column])
        y_val = val_data[target_column]

        # Get model class and parameters
        model_class = self._get_model_class(model_type)
        model_params = self._get_model_params(model_type, custom_params)

        # Train model
        model = model_class(**model_params)
        model.fit(X_train, y_train)

        # Evaluate
        predictions = model.predict(X_val)
        metrics = self._calculate_metrics(y_val, predictions)

        return model, metrics

    def _get_model_class(self, model_type: str):
        from sklearn import (  # type: ignore
            ensemble,
            linear_model,
            tree
        )

        model_map = {
            'random_forest': ensemble.RandomForestClassifier,
            'gradient_boosting': ensemble.GradientBoostingClassifier,
            'logistic_regression': linear_model.LogisticRegression,
            'decision_tree': tree.DecisionTreeClassifier
        }

        if model_type not in model_map:
            raise ValueError(f"Unsupported model type: {model_type}")

        return model_map[model_type]

    def _get_model_params(
        self,
        model_type: str,
        custom_params: Optional[Dict]
    ) -> Dict:
        default_params = self.training_config['frameworks_config']['sklearn']['default_settings']
        if custom_params:
            default_params.update(custom_params)
        return default_params

    def _calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series
    ) -> Dict:
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }

    def _save_model(self, model, model_type: str) -> Path:
        save_path = Path(self.model_config['model_paths']['saved_models'])
        save_path.mkdir(parents=True, exist_ok=True)

        model_path = save_path / f"{model_type}_model.joblib"
        joblib.dump(model, model_path)

        return model_path

    def _upload_model(
        self,
        model_path: Path,
        metrics: Dict,
        model_type: str
    ) -> Dict:
        # Prepare model for upload
        upload_dir = self.model_manager.prepare_model_upload(
            model_path.parent,
            framework="sklearn",
            include_artifacts=True
        )

        # Create metadata
        metadata = self.model_manager.create_metadata(
            name=f"{model_type}_model",
            version_name=f"v1.0",
            description=f"Trained {model_type} model",
            framework="sklearn",
            task_ids=["classification"],
            training_params=metrics
        )

        # Upload model
        return self.model_manager.upload_model(
            upload_dir,
            metadata,
            wait_for_processing=True
        )

def main():
    try:
        pipeline = ModelTrainingPipeline()

        # Train model
        result = pipeline.train_model(
            dataset_name="titanic",
            model_type="random_forest",
            target_column="Survived",
            custom_params={
                'n_estimators': 100,
                'max_depth': 10
            }
        )

        print("\nTraining Results:")
        print(f"Model saved to: {result['model_path']}")
        print("\nMetrics:")
        for metric, value in result['metrics'].items():
            print(f"{metric}: {value:.4f}")

        if result['upload_result']:
            print(f"\nModel Upload Result: {result['upload_result']}")

    except Exception as e:
        print(f"Error running pipeline: {str(e)}")

if __name__ == "__main__":
    main()
