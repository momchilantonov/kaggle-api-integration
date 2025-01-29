import pytest
from click.testing import CliRunner
import click
from unittest.mock import Mock, patch
from pathlib import Path
import json

# Import CLI commands
from cli.commands.competition_commands import competition
from cli.commands.dataset_commands import dataset
from cli.commands.model_commands import model
from cli.commands.kernel_commands import kernel

# Change to:
from src.cli.commands.competition_commands import competition
from src.cli.commands.dataset_commands import dataset
from src.cli.commands.model_commands import model
from src.cli.commands.kernel_commands import kernel

# The rest of the imports remain the same
from src.api.competitions import CompetitionClient
from src.api.datasets import DatasetClient
from src.api.models import ModelClient
from src.api.kernels import KernelClient
from src.utils.path_manager import PathManager
from src.handlers.data_handlers import DataHandler

@pytest.fixture
def cli_runner():
    return CliRunner()

@pytest.fixture
def mock_competition_client():
    with patch('src.api.competitions.CompetitionClient', autospec=True) as mock:
        instance = mock.return_value
        # Setup default return values
        instance.list_competitions.return_value = []
        instance.get_competition_details.return_value = {}
        yield instance

@pytest.fixture
def mock_dataset_client():
    with patch('src.api.datasets.DatasetClient', autospec=True) as mock:
        instance = mock.return_value
        instance.list_datasets.return_value = []
        instance.download_dataset.return_value = Path("/mock/path")
        yield instance

@pytest.fixture
def mock_model_client():
    with patch('src.api.models.ModelClient', autospec=True) as mock:
        instance = mock.return_value
        instance.list_models.return_value = []
        instance.pull_model.return_value = Path("/mock/path")
        yield instance

@pytest.fixture
def mock_kernel_client():
    with patch('src.api.kernels.KernelClient', autospec=True) as mock:
        instance = mock.return_value
        instance.list_kernels.return_value = []
        instance.get_kernel_status.return_value = {"status": "complete"}
        yield instance

@pytest.fixture
def mock_path_manager():
    with patch('src.utils.path_manager.PathManager', autospec=True) as mock:
        instance = mock.return_value
        instance.get_path.return_value = Path("/mock/path")
        yield instance

@pytest.fixture
def sample_competition_data():
    return [
        {
            "title": "Competition 1",
            "deadline": "2024-12-31",
            "reward": "$10,000",
            "category": "Featured"
        },
        {
            "title": "Competition 2",
            "deadline": "2024-12-31",
            "reward": "$5,000",
            "category": "Research"
        }
    ]

@pytest.fixture
def sample_dataset_data():
    return [
        {
            "title": "Dataset 1",
            "size": "1.2GB",
            "owner": "user1",
            "lastUpdated": "2024-01-01"
        },
        {
            "title": "Dataset 2",
            "size": "2.5GB",
            "owner": "user2",
            "lastUpdated": "2024-01-02"
        }
    ]

# Competition Command Tests
def test_competition_list(cli_runner, mock_competition_client, sample_competition_data):
    """Test competition list command"""
    mock_competition_client.list_competitions.return_value = sample_competition_data

    result = cli_runner.invoke(competition, ['list'])

    assert result.exit_code == 0
    assert 'Competition 1' in result.output
    assert 'Competition 2' in result.output
    assert '$10,000' in result.output
    mock_competition_client.list_competitions.assert_called_once()

def test_competition_download(cli_runner, mock_competition_client, tmp_path):
    """Test competition download command"""
    mock_competition_client.download_competition_files.return_value = tmp_path / "competition"

    result = cli_runner.invoke(competition, ['download', 'test-competition', '-o', str(tmp_path)])

    assert result.exit_code == 0
    assert 'Successfully downloaded' in result.output
    mock_competition_client.download_competition_files.assert_called_once_with(
        competition='test-competition',
        path=Path(tmp_path)
    )

def test_competition_submit(cli_runner, mock_competition_client, tmp_path):
    """Test competition submission command"""
    # Create a test submission file
    submission_file = tmp_path / "submission.csv"
    submission_file.touch()

    mock_competition_client.submit_to_competition.return_value = {
        "status": "success",
        "score": 0.95
    }

    result = cli_runner.invoke(competition, [
        'submit',
        'test-competition',
        str(submission_file),
        '-m', 'Test submission'
    ])

    assert result.exit_code == 0
    assert 'success' in result.output.lower()
    assert '0.95' in result.output
    mock_competition_client.submit_to_competition.assert_called_once()

# Dataset Command Tests
def test_dataset_list(cli_runner, mock_dataset_client, sample_dataset_data):
    """Test dataset list command"""
    mock_dataset_client.list_datasets.return_value = sample_dataset_data

    result = cli_runner.invoke(dataset, ['list', '--search', 'test'])

    assert result.exit_code == 0
    assert 'Dataset 1' in result.output
    assert 'Dataset 2' in result.output
    assert '1.2GB' in result.output
    mock_dataset_client.list_datasets.assert_called_once_with(search='test', page=1)

def test_dataset_download(cli_runner, mock_dataset_client, tmp_path):
    """Test dataset download command"""
    mock_dataset_client.download_dataset.return_value = tmp_path / "dataset"

    result = cli_runner.invoke(dataset, ['download', 'test/dataset', '-o', str(tmp_path)])

    assert result.exit_code == 0
    assert 'Successfully downloaded' in result.output
    mock_dataset_client.download_dataset.assert_called_once()

# Model Command Tests
def test_model_list(cli_runner, mock_model_client):
    """Test model list command"""
    mock_model_client.list_models.return_value = [
        {"name": "Model 1", "framework": "pytorch", "version": "1.0"},
        {"name": "Model 2", "framework": "tensorflow", "version": "2.0"}
    ]

    result = cli_runner.invoke(model, ['list'])

    assert result.exit_code == 0
    assert 'Model 1' in result.output
    assert 'pytorch' in result.output.lower()
    mock_model_client.list_models.assert_called_once()

def test_model_download(cli_runner, mock_model_client, tmp_path):
    """Test model download command"""
    mock_model_client.pull_model.return_value = tmp_path / "model"

    result = cli_runner.invoke(model, ['download', 'owner', 'model-name', '-o', str(tmp_path)])

    assert result.exit_code == 0
    assert 'Successfully downloaded' in result.output
    mock_model_client.pull_model.assert_called_once()

# Kernel Command Tests
def test_kernel_list(cli_runner, mock_kernel_client):
    """Test kernel list command"""
    mock_kernel_client.list_kernels.return_value = [
        {"title": "Kernel 1", "language": "python", "totalVotes": 10},
        {"title": "Kernel 2", "language": "r", "totalVotes": 5}
    ]

    result = cli_runner.invoke(kernel, ['list'])

    assert result.exit_code == 0
    assert 'Kernel 1' in result.output
    assert 'python' in result.output.lower()
    mock_kernel_client.list_kernels.assert_called_once()

def test_kernel_status(cli_runner, mock_kernel_client):
    """Test kernel status command"""
    mock_kernel_client.get_kernel_status.return_value = {
        "status": "complete",
        "error": None,
        "lastRunTime": "2024-01-01"
    }

    result = cli_runner.invoke(kernel, ['status', 'owner', 'kernel-name'])

    assert result.exit_code == 0
    assert 'complete' in result.output.lower()
    mock_kernel_client.get_kernel_status.assert_called_once()

# Error Handling Tests
def test_invalid_competition_submit(cli_runner, mock_competition_client):
    """Test invalid competition submission"""
    result = cli_runner.invoke(competition, [
        'submit',
        'test-competition',
        'nonexistent.csv',
        '-m', 'Test submission'
    ])

    assert result.exit_code != 0
    assert 'error' in result.output.lower()

def test_invalid_dataset_download(cli_runner, mock_dataset_client):
    """Test invalid dataset download"""
    mock_dataset_client.download_dataset.side_effect = click.exceptions.Abort()

    result = cli_runner.invoke(dataset, ['download', 'invalid/dataset'])

    assert result.exit_code != 0
    assert 'error' in result.output.lower()

def test_competition_submit_without_message(cli_runner):
    """Test competition submission without required message"""
    result = cli_runner.invoke(competition, [
        'submit',
        'test-competition',
        'submission.csv'
    ])

    assert result.exit_code != 0
    assert 'message' in result.output.lower()

def test_model_download_with_version(cli_runner, mock_model_client, tmp_path):
    """Test model download with specific version"""
    mock_model_client.pull_model.return_value = tmp_path / "model"

    result = cli_runner.invoke(model, [
        'download',
        'owner',
        'model-name',
        '--version', 'v1.0',
        '-o', str(tmp_path)
    ])

    assert result.exit_code == 0
    mock_model_client.pull_model.assert_called_once()
