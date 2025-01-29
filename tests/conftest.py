import pytest
from pathlib import Path
import json
import shutil
import tempfile
from unittest.mock import Mock
import os

from src.api.kaggle_client import KaggleAPIClient
from src.utils.path_manager import PathManager
from src.handlers.data_handlers import DataHandler


# Add this setup function
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup function to initialize test environment"""
    # Create temporary test directory
    test_dir = tempfile.mkdtemp()
    test_base_path = Path(test_dir)

    # Create necessary test directories
    test_directories = [
        'data/competitions',
        'data/datasets',
        'data/models',
        'data/kernels',
        'data/files',
        'logs'
    ]

    for directory in test_directories:
        (test_base_path / directory).mkdir(parents=True, exist_ok=True)

    # Set up environment variables for testing
    os.environ['KAGGLE_USERNAME'] = 'test_user'
    os.environ['KAGGLE_KEY'] = 'test_key'

    yield test_base_path

    # Cleanup after all tests
    try:
        shutil.rmtree(test_dir)
    except Exception as e:
        print(f"Error cleaning up test directory: {e}")

    # Reset environment variables
    if 'KAGGLE_USERNAME' in os.environ:
        del os.environ['KAGGLE_USERNAME']
    if 'KAGGLE_KEY' in os.environ:
        del os.environ['KAGGLE_KEY']

@pytest.fixture
def mock_kaggle_client():
    """Fixture to create a mock KaggleAPIClient"""
    mock_request_manager = Mock()
    client = Mock()
    client.credentials = {'username': 'test_user', 'key': 'test_key'}
    client.request_manager = mock_request_manager
    return client

@pytest.fixture
def test_path_manager():
    """Fixture to create a PathManager with test directories"""
    test_base_path = Path('test_data')
    path_manager = PathManager(base_path=test_base_path)
    path_manager.ensure_directories()
    yield path_manager
    # Cleanup test directories after tests
    if test_base_path.exists():
        import shutil
        shutil.rmtree(test_base_path)

@pytest.fixture
def sample_csv_data():
    """Fixture to provide sample CSV data"""
    return {
        'id': range(1, 6),
        'value': [10.0, 20.0, None, 40.0, 50.0],
        'category': ['A', 'B', 'A', None, 'B']
    }

@pytest.fixture
def sample_dataframe(sample_csv_data):
    """Fixture to provide a sample pandas DataFrame"""
    import pandas as pd
    return pd.DataFrame(sample_csv_data)

@pytest.fixture
def sample_csv_path(tmp_path, sample_dataframe):
    """Fixture to create a sample CSV file"""
    file_path = tmp_path / "test.csv"
    sample_dataframe.to_csv(file_path, index=False)
    return file_path

@pytest.fixture
def mock_env_vars():
    """Fixture to set up test environment variables"""
    original_env = dict(os.environ)
    os.environ.update({
        'KAGGLE_USERNAME': 'test_user',
        'KAGGLE_KEY': 'test_key'
    })
    yield
    os.environ.clear()
    os.environ.update(original_env)

@pytest.fixture
def test_data_handler():
    """Fixture to provide a DataHandler instance"""
    return DataHandler()

@pytest.fixture
def sample_config_data():
    """Fixture to provide sample configuration data"""
    return {
        'competition_params': {
            'data_paths': {
                'default': 'data/competitions',
                'submissions': 'data/competitions/submissions',
                'leaderboards': 'data/competitions/leaderboards'
            },
            'active_competitions': {
                'test_competition': {
                    'deadline': '2024-12-31',
                    'metric': 'accuracy',
                    'file_structure': {
                        'train': 'train.csv',
                        'test': 'test.csv'
                    }
                }
            }
        }
    }

@pytest.fixture
def sample_kaggle_response():
    """Fixture to provide sample Kaggle API response"""
    class MockResponse:
        def __init__(self, status_code=200, json_data=None):
            self.status_code = status_code
            self._json_data = json_data or {}
            self.content = json.dumps(self._json_data).encode()

        def json(self):
            return self._json_data

        def raise_for_status(self):
            if self.status_code >= 400:
                from requests.exceptions import HTTPError
                raise HTTPError(f"HTTP Error: {self.status_code}")

    return MockResponse

@pytest.fixture
def mock_requests_session(monkeypatch):
    """Fixture to mock requests session"""
    mock_session = Mock()
    mock_session.auth = ('test_user', 'test_key')

    def mock_request(*args, **kwargs):
        return Mock(status_code=200, json=lambda: {'success': True})

    mock_session.request = mock_request
    monkeypatch.setattr('requests.Session', lambda: mock_session)
    return mock_session

@pytest.fixture
def sample_model_data():
    """Fixture to provide sample model data"""
    return {
        'model_type': 'random_forest',
        'parameters': {
            'n_estimators': 100,
            'max_depth': 10
        },
        'metrics': {
            'accuracy': 0.85,
            'precision': 0.83,
            'recall': 0.82,
            'f1': 0.84
        }
    }

@pytest.fixture
def sample_workflow_data():
    """Fixture to provide sample workflow data"""
    return {
        'dataset': {
            'name': 'test_dataset',
            'files': ['train.csv', 'test.csv'],
            'process_config': {
                'handle_missing': True,
                'missing_strategy': {
                    'numeric': 'mean',
                    'categorical': 'mode'
                }
            }
        },
        'model': {
            'type': 'random_forest',
            'target_column': 'target',
            'params': {
                'n_estimators': 100
            }
        }
    }
