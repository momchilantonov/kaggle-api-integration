import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import requests
import click

from src.api.models import ModelClient, ModelMetadata

@pytest.fixture
def model_client(mock_kaggle_client):
    return ModelClient(mock_kaggle_client)

@pytest.fixture
def sample_model_metadata():
    return ModelMetadata(
        name="test-model",
        version_name="v1.0",
        description="Test model description",
        framework="pytorch",
        task_ids=["image-classification"],
        training_data="test/dataset",
        model_type="cnn",
        training_params={"epochs": 10, "batch_size": 32},
        license="MIT"
    )

def test_model_client_initialization(model_client):
    """Test model client initialization"""
    assert hasattr(model_client, 'client')

def test_list_models(model_client):
    """Test listing models"""
    mock_response = Mock()
    mock_response.json.return_value = [
        {"name": "Model 1", "framework": "pytorch"},
        {"name": "Model 2", "framework": "tensorflow"}
    ]
    model_client.client.get.return_value = mock_response

    result = model_client.list_models(owner="test-user", search="classification")

    model_client.client.get.assert_called_once_with(
        'models/list',
        params={'owner': 'test-user', 'search': 'classification', 'page': 1}
    )
    assert len(result) == 2
    assert result[0]["name"] == "Model 1"

def test_pull_model(model_client, tmp_path):
    """Test pulling model"""
    mock_response = Mock()
    mock_response.url = "http://example.com/model.zip"
    model_client.client.get.return_value = mock_response

    download_path = tmp_path / "model.zip"
    model_client.client.download_file.return_value = download_path

    result = model_client.pull_model(
        owner="test-user",
        model_name="test-model",
        path=tmp_path
    )

    assert result == download_path
    model_client.client.get.assert_called_once()

def test_push_model(model_client, sample_model_metadata, tmp_path):
    """Test pushing model"""
    # Create test model folder with files
    model_path = tmp_path / "test_model"
    model_path.mkdir()
    (model_path / "model.pt").touch()

    # Mock responses
    init_response = Mock()
    init_response.json.return_value = {"id": "model123"}
    upload_response = Mock()
    upload_response.json.return_value = {"success": True}
    push_response = Mock()
    push_response.json.return_value = {"version": "1.0"}

    model_client.client.post.side_effect = [init_response, upload_response, push_response]

    result = model_client.push_model(
        path=model_path,
        metadata=sample_model_metadata,
        version_notes="Initial version"
    )

    assert result["version"] == "1.0"
    assert model_client.client.post.call_count >= 1

def test_get_model_status(model_client):
    """Test getting model status"""
    mock_response = Mock()
    mock_response.json.return_value = {
        "status": "complete",
        "error": None
    }
    model_client.client.get.return_value = mock_response

    result = model_client.get_model_status(
        owner="test-user",
        model_name="test-model"
    )

    assert result["status"] == "complete"

def test_wait_for_model_ready(model_client):
    """Test waiting for model to be ready"""
    # Mock multiple status checks
    mock_responses = [
        {"status": "processing"},
        {"status": "processing"},
        {"status": "complete"}
    ]

    model_client.get_model_status = Mock(side_effect=mock_responses)

    with patch('time.sleep') as mock_sleep:
        result = model_client.wait_for_model_ready(
            owner="test-user",
            model_name="test-model"
        )

    assert result["status"] == "complete"

def test_model_metadata_conversion(sample_model_metadata):
    """Test model metadata conversion to dict"""
    metadata_dict = sample_model_metadata.to_dict()

    assert metadata_dict['name'] == "test-model"
    assert metadata_dict['version_name'] == "v1.0"
    assert metadata_dict['framework'] == "pytorch"
    assert metadata_dict['task_ids'] == ["image-classification"]
    assert metadata_dict['training_params'] == {"epochs": 10, "batch_size": 32}

def test_invalid_model_pull(model_client):
    """Test pulling non-existent model"""
    mock_response = Mock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        response=mock_response
    )
    model_client.client.get.side_effect = requests.exceptions.HTTPError(
        response=mock_response
    )

    with pytest.raises(click.exceptions.Abort):
        model_client.pull_model(
            owner="invalid",
            model_name="nonexistent"
        )

def test_model_timeout(model_client):
    """Test model readiness timeout"""
    mock_response = Mock()
    mock_response.json.return_value = {"status": "processing"}
    model_client.client.get.return_value = mock_response

    with patch('time.sleep') as mock_sleep:
        with pytest.raises(TimeoutError):
            model_client.wait_for_model_ready(
                owner="test-user",
                model_name="test-model",
                timeout=1
            )

def test_model_error_status(model_client):
    """Test model error status"""
    mock_response = Mock()
    mock_response.status_code = 400
    mock_response.json.return_value = {
        "status": "error",
        "errorMessage": "Processing failed"
    }
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        response=mock_response
    )
    model_client.client.get.side_effect = requests.exceptions.HTTPError(
        response=mock_response
    )

    with pytest.raises(click.exceptions.Abort):
        model_client.get_model_status(
            owner="test-user",
            model_name="test-model"
        )
