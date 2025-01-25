import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import json
import time

from src.api.kaggle_client import KaggleAPIClient
from src.api.kernels import KernelClient, KernelMetadata

@pytest.fixture
def mock_kaggle_client():
    """Fixture to create a mock KaggleAPIClient"""
    return Mock(spec=KaggleAPIClient)

@pytest.fixture
def kernel_client(mock_kaggle_client):
    """Fixture to create a KernelClient with mock KaggleAPIClient"""
    return KernelClient(mock_kaggle_client)

@pytest.fixture
def sample_metadata():
    """Fixture to create sample kernel metadata"""
    return KernelMetadata(
        title="Test Kernel",
        language="python",
        kernel_type="notebook",
        is_private=False,
        enable_gpu=True,
        enable_internet=True,
        dataset_sources=["user/dataset1"],
        competition_sources=["competition1"],
        kernel_sources=["user/kernel1"]
    )

def test_list_kernels(kernel_client, mock_kaggle_client):
    """Test listing kernels with filters"""
    mock_response = Mock()
    mock_response.json.return_value = [
        {"title": "Kernel1", "language": "python"},
        {"title": "Kernel2", "language": "r"}
    ]
    mock_kaggle_client.get.return_value = mock_response

    kernels = kernel_client.list_kernels(
        owner="testuser",
        language="python",
        kernel_type="notebook",
        page=2
    )

    mock_kaggle_client.get.assert_called_once()
    call_params = mock_kaggle_client.get.call_args[1]['params']
    assert call_params['owner'] == "testuser"
    assert call_params['language'] == "python"
    assert call_params['kernelType'] == "notebook"
    assert call_params['page'] == 2
    assert len(kernels) == 2

def test_push_kernel(kernel_client, mock_kaggle_client, tmp_path, sample_metadata):
    """Test pushing a kernel"""
    # Create test kernel files
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    (kernel_dir / "notebook.ipynb").write_text("notebook content")
    (kernel_dir / "data.csv").write_text("data content")

    mock_response = Mock()
    mock_response.json.return_value = {"status": "success"}
    mock_kaggle_client.post.return_value = mock_response

    result = kernel_client.push_kernel(kernel_dir, sample_metadata)

    # Check metadata push
    mock_kaggle_client.post.assert_called_once_with(
        'kernel_push',
        json=sample_metadata.to_dict()
    )

    # Check file uploads
    assert mock_kaggle_client.upload_file.call_count == 2
    assert result == {"status": "success"}

def test_push_kernel_invalid_path(kernel_client, sample_metadata, tmp_path):
    """Test pushing kernel with invalid path"""
    invalid_path = tmp_path / "nonexistent"

    with pytest.raises(FileNotFoundError):
        kernel_client.push_kernel(invalid_path, sample_metadata)

def test_pull_kernel(kernel_client, mock_kaggle_client, tmp_path):
    """Test pulling a kernel"""
    mock_response = Mock()
    mock_response.json.return_value = [
        {"path": "notebook.ipynb", "source": "notebook content"},
        {"path": "data/data.csv", "source": "data content"}
    ]
    mock_kaggle_client.get.return_value = mock_response

    result = kernel_client.pull_kernel(
        "owner",
        "kernel",
        "v1",
        tmp_path
    )

    mock_kaggle_client.get.assert_called_once_with(
        'kernel_pull',
        params={
            'ownerSlug': 'owner',
            'kernelSlug': 'kernel',
            'version': 'v1'
        }
    )

    assert result == tmp_path / "kernel"
    assert (result / "notebook.ipynb").exists()
    assert (result / "data" / "data.csv").exists()

def test_get_kernel_status(kernel_client, mock_kaggle_client):
    """Test getting kernel status"""
    mock_response = Mock()
    mock_response.json.return_value = {
        "status": "complete",
        "version": "v1"
    }
    mock_kaggle_client.get.return_value = mock_response

    status = kernel_client.get_kernel_status("owner", "kernel", "v1")

    mock_kaggle_client.get.assert_called_once_with(
        'kernel_status',
        params={
            'ownerSlug': 'owner',
            'kernelSlug': 'kernel',
            'version': 'v1'
        }
    )
    assert status['status'] == "complete"

def test_wait_for_kernel_output_success(kernel_client, mock_kaggle_client):
    """Test waiting for kernel output - success case"""
    status_response = Mock()
    status_response.json.return_value = {"status": "complete"}

    output_response = Mock()
    output_response.json.return_value = {"output": "test output"}

    mock_kaggle_client.get.side_effect = [status_response, output_response]

    with patch('time.sleep'):  # Avoid actual sleeping in tests
        result = kernel_client.wait_for_kernel_output(
            "owner",
            "kernel",
            check_interval=1
        )

    assert result == {"output": "test output"}
    assert mock_kaggle_client.get.call_count == 2

def test_wait_for_kernel_output_error(kernel_client, mock_kaggle_client):
    """Test waiting for kernel output - error case"""
    mock_response = Mock()
    mock_response.json.return_value = {
        "status": "error",
        "errorMessage": "Test error"
    }
    mock_kaggle_client.get.return_value = mock_response

    with patch('time.sleep'):
        with pytest.raises(RuntimeError) as exc_info:
            kernel_client.wait_for_kernel_output("owner", "kernel")

        assert "Test error" in str(exc_info.value)

def test_wait_for_kernel_output_timeout(kernel_client, mock_kaggle_client):
    """Test timeout while waiting for kernel output"""
    mock_response = Mock()
    mock_response.json.return_value = {"status": "running"}
    mock_kaggle_client.get.return_value = mock_response

    with patch('time.sleep'):
        with pytest.raises(TimeoutError):
            kernel_client.wait_for_kernel_output(
                "owner",
                "kernel",
                timeout=1,
                check_interval=0.1
            )

def test_list_kernel_versions(kernel_client, mock_kaggle_client):
    """Test listing kernel versions"""
    mock_response = Mock()
    mock_response.json.return_value = [
        {"version": 1, "created": "2024-01-01"},
        {"version": 2, "created": "2024-01-02"}
    ]
    mock_kaggle_client.get.return_value = mock_response

    versions = kernel_client.list_kernel_versions("owner", "kernel")

    mock_kaggle_client.get.assert_called_once_with(
        'kernel_versions',
        params={
            'ownerSlug': 'owner',
            'kernelSlug': 'kernel'
        }
    )
    assert len(versions) == 2
    assert versions[0]['version'] == 1
    assert versions[1]['version'] == 2

def test_kernel_metadata_to_dict(sample_metadata):
    """Test KernelMetadata to_dict method"""
    metadata_dict = sample_metadata.to_dict()

    assert metadata_dict['title'] == "Test Kernel"
    assert metadata_dict['language'] == "python"
    assert metadata_dict['kernel_type'] == "notebook"
    assert metadata_dict['enable_gpu'] is True
    assert metadata_dict['dataset_sources'] == ["user/dataset1"]
    assert metadata_dict['competition_sources'] == ["competition1"]
