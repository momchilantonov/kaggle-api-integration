import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import requests
import click

from src.api.kernels import KernelClient, KernelMetadata

@pytest.fixture
def kernel_client(mock_kaggle_client):
    return KernelClient(mock_kaggle_client)

@pytest.fixture
def sample_kernel_metadata():
    return KernelMetadata(
        title="Test Kernel",
        language="python",
        kernel_type="script",
        is_private=False,
        enable_gpu=False,
        enable_internet=True,
        dataset_sources=["test/dataset"],
        competition_sources=None
    )

def test_kernel_client_initialization(kernel_client):
    """Test kernel client initialization"""
    assert hasattr(kernel_client, 'client')

def test_list_kernels(kernel_client):
    """Test listing kernels"""
    mock_response = Mock()
    mock_response.json.return_value = [
        {"title": "Kernel 1", "language": "python"},
        {"title": "Kernel 2", "language": "r"}
    ]
    kernel_client.client.get.return_value = mock_response

    result = kernel_client.list_kernels(owner="test-user", language="python")

    kernel_client.client.get.assert_called_once_with(
        'kernels/list',
        params={'owner': 'test-user', 'language': 'python', 'page': 1}
    )
    assert len(result) == 2
    assert result[0]["title"] == "Kernel 1"

def test_push_kernel(kernel_client, sample_kernel_metadata, tmp_path):
    """Test pushing kernel"""
    # Create test kernel folder
    kernel_folder = tmp_path / "test_kernel"
    kernel_folder.mkdir()
    (kernel_folder / "script.py").touch()

    # Mock responses
    mock_response = Mock()
    mock_response.json.return_value = {"id": "kernel123"}
    kernel_client.client.post.return_value = mock_response

    result = kernel_client.push_kernel(
        folder_path=kernel_folder,
        metadata=sample_kernel_metadata,
        version_notes="Initial version"
    )

    assert result["id"] == "kernel123"
    assert kernel_client.client.post.called

def test_pull_kernel(kernel_client, tmp_path):
    """Test pulling kernel"""
    mock_response = Mock()
    mock_response.json.return_value = [
        {"path": "script.py", "source": "print('Hello')"}
    ]
    kernel_client.client.get.return_value = mock_response

    result = kernel_client.pull_kernel(
        owner="test-user",
        kernel_name="test-kernel",
        path=tmp_path
    )

    assert result == tmp_path
    assert (tmp_path / "script.py").exists()

def test_get_kernel_status(kernel_client):
    """Test getting kernel status"""
    mock_response = Mock()
    mock_response.json.return_value = {
        "status": "complete",
        "error": None
    }
    kernel_client.client.get.return_value = mock_response

    result = kernel_client.get_kernel_status(
        owner="test-user",
        kernel_name="test-kernel"
    )

    assert result["status"] == "complete"

def test_wait_for_kernel_output(kernel_client):
    """Test waiting for kernel output"""
    # Mock status and output responses
    status_response = Mock()
    status_response.json.return_value = {"status": "complete"}

    output_response = Mock()
    output_response.json.return_value = {"output": "Test output"}

    kernel_client.client.get.side_effect = [status_response, output_response]

    with patch('time.sleep') as mock_sleep:
        result = kernel_client.wait_for_kernel_output(
            owner="test-user",
            kernel_name="test-kernel"
        )

    assert result["output"] == "Test output"

def test_kernel_error_status(kernel_client):
    """Test kernel error status"""
    mock_response = Mock()
    mock_response.status_code = 400  # Bad Request
    mock_response.json.return_value = {
        "status": "error",
        "errorMessage": "Runtime error"
    }
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        response=mock_response
    )
    kernel_client.client.get.side_effect = requests.exceptions.HTTPError(
        response=mock_response
    )

    with pytest.raises(click.exceptions.Abort):
        kernel_client.get_kernel_status(
            owner="test-user",
            kernel_name="test-kernel"
        )

def test_kernel_metadata_conversion(sample_kernel_metadata):
    """Test kernel metadata conversion to dict"""
    metadata_dict = sample_kernel_metadata.to_dict()

    assert metadata_dict['title'] == "Test Kernel"
    assert metadata_dict['language'] == "python"
    assert metadata_dict['kernel_type'] == "script"
    assert metadata_dict['is_private'] == False
    assert metadata_dict['enable_gpu'] == False
    assert metadata_dict['enable_internet'] == True
    assert metadata_dict['dataset_sources'] == ["test/dataset"]

def test_invalid_kernel_pull(kernel_client):
    """Test pulling non-existent kernel"""
    mock_response = Mock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        response=mock_response
    )
    kernel_client.client.get.side_effect = requests.exceptions.HTTPError(
        response=mock_response
    )

    with pytest.raises(click.exceptions.Abort):
        kernel_client.pull_kernel(
            owner="invalid",
            kernel_name="nonexistent"
        )

def test_kernel_output_timeout(kernel_client):
    """Test kernel output timeout"""
    mock_response = Mock()
    mock_response.json.return_value = {"status": "running"}
    kernel_client.client.get.return_value = mock_response

    with patch('time.sleep') as mock_sleep:
        with pytest.raises(TimeoutError):
            kernel_client.wait_for_kernel_output(
                owner="test-user",
                kernel_name="test-kernel",
                timeout=1
            )
