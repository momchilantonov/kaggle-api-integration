from pathlib import Path
import yaml
import logging
from typing import Optional, Dict, List, Union, Callable
import pandas as pd
import json
from datetime import datetime

from workflows.dataset_workflows.download_operations import DatasetDownloadManager
from workflows.model_workflows.upload_operations import ModelUploadManager
from src.handlers.data_handlers import DataHandler
from src.utils.path_manager import PathManager
from src.utils.error_handlers import handle_api_errors
from src.utils.helpers import timer

logger = logging.getLogger(__name__)

class ModelTrainingPipeline:
    """Pipeline for model training and evaluation"""

    def __init__(self):
        self.dataset_manager = DatasetDownloadManager()
        self.model_manager = ModelUploadManager()
        self.data_handler = DataHandler()
        self.path_manager = PathManager()
        # Ensure required directories exist
        self.path_manager.ensure_directories()
        self._load_configs()

    def _load_configs(self):
        """Load training and model configurations"""
        try:
            model_config_path = Path('operational_configs/model_configs')

            with open(model_config_path / 'training_config.yaml', 'r') as f:
                self.training_config = yaml.safe_load(f)

            with open(model_config_path / 'model_params.yaml', 'r') as f:
                self.model_config = yaml.safe_load(f)

            logger.info("Successfully loaded training configurations")

        except Exception as e:
            logger.error(f"Error loading configurations: {str(e)}")
            raise

    @timer
    @handle_api_errors
    def train_model(
        self,
        dataset_name: str,
        model_type: str,
        target_column: str,
        custom_params: Optional[Dict] = None,
        callback: Optional[Callable] = None
    ) -> Dict:
        """Train a model with specified configuration"""
        try:
            if callback:
                callback(0, "Starting model training pipeline")

            # Prepare data
            train_data, val_data = self._prepare_data(
                dataset_name,
                target_column
            )
            if callback:
                callback(30, "Data prepared")

            # Train model
            model, metrics = self._train_and_evaluate(
                train_data,
                val_data,
                target_column,
                model_type,
                custom_params
            )
            if callback:
                callback(70, "Model trained and evaluated")

            # Save model
            model_path = self._save_model(
                model,
                model_type,
                metrics
            )
            if callback:
                callback(90, "Model saved")

            # Upload if configured
            upload_result = None
            if self.model_config['model_settings'].get('auto_upload', False):
                upload_result = self._upload_model(
                    model_path,
                    metrics,
                    model_type
                )
                if callback:
                    callback(100, "Model uploaded")

            result = {
                'model_path': str(model_path),
                'metrics': metrics,
                'upload_result': upload_result
            }

            # Log training execution
            self._log_training_execution(result)

            return result

        except Exception as e:
            logger.error(f"Error in model training pipeline: {str(e)}")
            raise

    def _prepare_data(
        self,
        dataset_name: str,
        target_column: str
    ) -> tuple:
        """Prepare and split data for training"""
        try:
            # Download and load data
            dataset_path = self.dataset_manager.download_dataset(dataset_name)
            data = self.data_handler.read_csv(dataset_path / "train.csv")

            # Apply preprocessing
            preprocessing_config = self.training_config['data_handling']['preprocessing']

            # Handle missing values
            data = self.data_handler.handle_missing_values(
                data,
                preprocessing_config['handle_missing']
            )

            # Apply data type conversions
            if preprocessing_config.get('data_types'):
                data = self.data_handler.convert_dtypes(
                    data,
                    preprocessing_config['data_types']
                )

            # Apply preprocessing steps
            data = self._apply_preprocessing(data, preprocessing_config)

            # Split data
            train_size = self.training_config['training_settings']['default']['validation_split']
            train_data, val_data = self.data_handler.split_dataset(
                data,
                train_size=train_size,
                random_state=42
            )

            return train_data, val_data

        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise

    def _apply_preprocessing(
        self,
        data: pd.DataFrame,
        config: Dict
    ) -> pd.DataFrame:
        """Apply preprocessing steps to data"""
        try:
            # Apply normalization if specified
            if config.get('normalization'):
                for col in data.select_dtypes(include=['float64', 'int64']).columns:
                    if config['normalization'] == 'standard':
                        data[col] = (data[col] - data[col].mean()) / data[col].std()
                    elif config['normalization'] == 'minmax':
                        data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())

            # Handle categorical encoding
            if config.get('categorical_encoding') == 'onehot':
                categorical_columns = data.select_dtypes(include=['object']).columns
                for col in categorical_columns:
                    dummies = pd.get_dummies(data[col], prefix=col)
                    data = pd.concat([data, dummies], axis=1)
                    data.drop(columns=[col], inplace=True)

            # Text preprocessing if specified
            if config.get('text_preprocessing'):
                text_config = config['text_preprocessing']
                text_columns = data.select_dtypes(include=['object']).columns
                for col in text_columns:
                    if text_config.get('lowercase'):
                        data[col] = data[col].str.lower()
                    if text_config.get('remove_punctuation'):
                        data[col] = data[col].str.replace(r'[^\w\s]', '')
                    if text_config.get('remove_numbers'):
                        data[col] = data[col].str.replace(r'\d+', '')

            return data

        except Exception as e:
            logger.error(f"Error applying preprocessing: {str(e)}")
            raise

    def _train_and_evaluate(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        target_column: str,
        model_type: str,
        custom_params: Optional[Dict]
    ) -> tuple:
        """Train and evaluate model"""
        try:
            # Prepare features and target
            X_train = train_data.drop(columns=[target_column])
            y_train = train_data[target_column]
            X_val = val_data.drop(columns=[target_column])
            y_val = val_data[target_column]

            # Get model class and parameters
            model_class = self._get_model_class(model_type)
            model_params = self._get_model_params(model_type, custom_params)

            # Initialize model with parameters
            model = model_class(**model_params)

            # Train model with monitoring
            self._monitor_training_start(model_type)
            model.fit(X_train, y_train)
            training_time = self._monitor_training_end()

            # Evaluate model
            metrics = self._evaluate_model(
                model,
                X_val,
                y_val,
                model_type
            )

            # Add training time to metrics
            metrics['training_time'] = training_time

            return model, metrics

        except Exception as e:
            logger.error(f"Error training and evaluating model: {str(e)}")
            raise

    def _get_model_class(self, model_type: str):
        """Get appropriate model class based on type"""
        try:
            from sklearn import (
                ensemble,
                linear_model,
                tree,
                svm,
                neighbors
            )

            model_map = {
                'random_forest': ensemble.RandomForestClassifier,
                'gradient_boosting': ensemble.GradientBoostingClassifier,
                'logistic_regression': linear_model.LogisticRegression,
                'decision_tree': tree.DecisionTreeClassifier,
                'svm': svm.SVC,
                'knn': neighbors.KNeighborsClassifier
            }

            if model_type not in model_map:
                raise ValueError(f"Unsupported model type: {model_type}")

            return model_map[model_type]

        except Exception as e:
            logger.error(f"Error getting model class: {str(e)}")
            raise

    def _get_model_params(
        self,
        model_type: str,
        custom_params: Optional[Dict]
    ) -> Dict:
        """Get model parameters"""
        try:
            # Get default parameters for model type
            default_params = self.training_config['frameworks_config']['sklearn']['default_settings']

            # Get model-specific parameters if available
            model_specific_params = self.model_config['model_types'].get(
                model_type,
                {}
            ).get('default_params', {})

            # Combine parameters with custom ones taking precedence
            params = {
                **default_params,
                **model_specific_params,
                **(custom_params or {})
            }

            return params

        except Exception as e:
            logger.error(f"Error getting model parameters: {str(e)}")
            raise

    def _monitor_training_start(self, model_type: str) -> None:
        """Start monitoring training process"""
        try:
            self.training_start_time = datetime.now()

            log_entry = {
                'timestamp': self.training_start_time.isoformat(),
                'model_type': model_type,
                'event': 'training_start'
            }

            self._append_to_training_log(log_entry)

        except Exception as e:
            logger.error(f"Error starting training monitoring: {str(e)}")

    def _monitor_training_end(self) -> float:
        """End monitoring training process and return duration"""
        try:
            end_time = datetime.now()
            duration = (end_time - self.training_start_time).total_seconds()

            log_entry = {
                'timestamp': end_time.isoformat(),
                'event': 'training_end',
                'duration_seconds': duration
            }

            self._append_to_training_log(log_entry)
            return duration

        except Exception as e:
            logger.error(f"Error ending training monitoring: {str(e)}")
            return 0.0

    def _evaluate_model(
        self,
        model,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame,
        model_type: str
    ) -> Dict:
        """Evaluate model performance"""
        try:
            from sklearn import metrics

            # Make predictions
            y_pred = model.predict(X_val)

            # Calculate metrics based on problem type
            if self._is_classification(model_type):
                evaluation = {
                    'accuracy': metrics.accuracy_score(y_val, y_pred),
                    'precision': metrics.precision_score(y_val, y_pred, average='weighted'),
                    'recall': metrics.recall_score(y_val, y_pred, average='weighted'),
                    'f1': metrics.f1_score(y_val, y_pred, average='weighted')
                }
            else:
                evaluation = {
                    'mse': metrics.mean_squared_error(y_val, y_pred),
                    'rmse': metrics.mean_squared_error(y_val, y_pred, squared=False),
                    'mae': metrics.mean_absolute_error(y_val, y_pred),
                    'r2': metrics.r2_score(y_val, y_pred)
                }

            return evaluation

        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise

    def _is_classification(self, model_type: str) -> bool:
        """Determine if model is for classification"""
        classification_models = {
            'random_forest',
            'logistic_regression',
            'decision_tree',
            'gradient_boosting',
            'svm',
            'knn'
        }
        return model_type in classification_models

    def _save_model(
        self,
        model,
        model_type: str,
        metrics: Dict
    ) -> Path:
        """Save trained model and metadata"""
        try:
            import joblib

            # Create model directory
            model_dir = self.path_manager.get_path('models', 'custom')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = model_dir / f"{model_type}_{timestamp}.joblib"

            # Save model
            joblib.dump(model, model_path)

            # Save metadata
            metadata = {
                'model_type': model_type,
                'timestamp': timestamp,
                'metrics': metrics,
                'parameters': model.get_params()
            }

            metadata_path = model_path.with_suffix('.meta.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Saved model and metadata to {model_path}")
            return model_path

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def _upload_model(
        self,
        model_path: Path,
        metrics: Dict,
        model_type: str
    ) -> Dict:
        """Upload model to Kaggle"""
        try:
            # Prepare model for upload
            upload_dir = self.model_manager.prepare_model_upload(
                model_path.parent,
                framework="sklearn",
                include_artifacts=True
            )

            # Create metadata
            metadata = self.model_manager.create_metadata(
                name=f"{model_type}_model",
                version_name="v1.0",
                description=self._generate_model_description(model_type, metrics),
                framework="sklearn",
                task_ids=["classification" if self._is_classification(model_type) else "regression"],
                training_params=metrics
            )

            # Upload model
            return self.model_manager.upload_model(
                upload_dir,
                metadata,
                wait_for_processing=True
            )

        except Exception as e:
            logger.error(f"Error uploading model: {str(e)}")
            raise

    def _generate_model_description(self, model_type: str, metrics: Dict) -> str:
        """Generate model description"""
        description = (
            f"# {model_type.replace('_', ' ').title()} Model\n\n"
            f"Trained on {datetime.now().strftime('%Y-%m-%d')}\n\n"
            f"## Model Performance\n"
        )

        for metric, value in metrics.items():
            if isinstance(value, float):
                description += f"- {metric}: {value:.4f}\n"
            else:
                description += f"- {metric}: {value}\n"

        return description

    def _append_to_training_log(self, log_entry: Dict) -> None:
        """Append entry to training log"""
        try:
            log_path = self.path_manager.get_path('logs') / 'model_training.log'

            with open(log_path, 'a') as f:
                f.write(f"{json.dumps(log_entry)}\n")

        except Exception as e:
            logger.error(f"Error appending to training log: {str(e)}")

    def _log_training_execution(self, result: Dict) -> None:
        """Log training execution details"""
        try:
            log_dir = self.path_manager.get_path('logs')
            log_path = log_dir / 'training_executions.log'

            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'execution_result': result
            }

            with open(log_path, 'a') as f:
                f.write(f"{json.dumps(log_entry)}\n")

        except Exception as e:
            logger.error(f"Error logging training execution: {str(e)}")

if __name__ == '__main__':
    # Example usage
    try:
        pipeline = ModelTrainingPipeline()

        def progress_callback(progress, status):
            print(f"Progress: {progress}%, Status: {status}")

        result = pipeline.train_model(
            dataset_name="titanic",
            model_type="random_forest",
            target_column="Survived",
            custom_params={
                'n_estimators': 100,
                'max_depth': 10
            },
            callback=progress_callback
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
