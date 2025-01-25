import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import time

from src.api.models import ModelClient, ModelMetadata
from src.api.kaggle_client import KaggleAPIClient

@pytest.fixture
def mock_kaggle_client():
    """Fixture to create a mock KaggleAPIClient"""
    return Mock(spec=KaggleAPIClient)

@pytest.fixture
def model_client(mock_kaggle_client):
    """Fixture to create a ModelClient with mock KaggleAPIClient"""
    return ModelClient(mock_kaggle_client)

@pytest.fixture
def sample_metadata():
    """Fixture to create sample model metadata"""
    return ModelMetadata(
        name="test-model",
        version_name="v1.0",
        description="Test model description",
        framework="PyTorch",
        task_ids=["computer-vision", "image-classification"],
        training_data="imagenet",
        model_type="resnet50",
        training_params={"epochs": 100, "batch_size": 32}
    )

def test_list_models(model_client, mock_kaggle_client):
    """Test listing models with filters"""
    mock_response = Mock()
    mock_response.json.return_value = [
        {"name": "model1", "framework": "PyTorch"},
        {"name": "model2", "framework": "TensorFlow"}
    ]
    mock_kaggle_client.get.return_value = mock_response

    models = model_client.list_models(
        owner="testuser",
        framework="PyTorch",
        task="computer-vision",
        page=2
    )

    mock_kaggle_client.get.assert_called_once()
    call_params = mock_kaggle_client.get.call_args[1]['params']
    assert call_params['owner'] == "testuser"
    assert call_params['framework'] == "PyTorch"
    assert call_params['task'] == "computer-vision"
    assert call_params['page'] == 2
    assert len(models) == 2

def test_pull_model(model_client, mock_kaggle_client, tmp_path):
    """Test pulling a model"""
    mock_response = Mock()
    mock_kaggle_client.get.return_value = mock_response
    mock_kaggle_client.download_file.return_value = tmp_path / "model.zip"

    result = model_client.pull_model(
        "owner",
        "model",
        "v1",
        tmp_path
    )

    mock_kaggle_client.get.assert_called_once_with(
        'model_pull',
        params={
            'ownerSlug': 'owner',
            'modelSlug': 'model',
            'versionNumber': 'v1'
        },
        stream=True
    )
    assert result == tmp_path / "model.zip"

def test_push_model(model_client, mock_kaggle_client, tmp_path, sample_metadata):
    """Test pushing a model"""
    # Create test model files
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.pt").write_text("model data")
    (model_dir / "config.json").write_text("config data")

    mock_init_response = Mock()
    mock_init_response.json.return_value = {"status": "initialized"}

    mock_push_response = Mock()
    mock_push_response.json.return_value = {"status": "success"}

    mock_kaggle_client.post.side_effect = [mock_init_response, mock_push_response]

    result = model_client.push_model(model_dir, sample_metadata)

    # Check model initialization
    mock_kaggle_client.post.assert_any_call(
        'model_initiate',
        json={
            **sample_metadata.to_dict(),
            'isPublic': True
        }
    )

    # Check file uploads
    assert mock_kaggle_client.upload_file.call_count == 2

    # Check final push
    mock_kaggle_client.post.assert_any_call(
        'model_push',
        json={'modelSlug': sample_metadata.name}
    )

    assert result == {"status": "success"}

def test_push_model_invalid_path(model_client, sample_metadata, tmp_path):
    """Test pushing model with invalid path"""
    invalid_path = tmp_path / "nonexistent"

    with pytest.raises(FileNotFoundError):
        model_client.push_model(invalid_path, sample_metadata)

def test_get_model_status(model_client, mock_kaggle_client):
    """Test getting model status"""
    mock_response = Mock()
    mock_response.json.return_value = {
        "status": "complete",
        "versionNumber": "v1"
    }
    mock_kaggle_client.get.return_value = mock_response

    status = model_client.get_model_status("owner", "model", "v1")

    mock_kaggle_client.get.assert_called_once_with(
        'model_status',
        params={
            'ownerSlug': 'owner',
            'modelSlug': 'model',
            'versionNumber': 'v1'
        }
    )
    assert status['status'] == "complete"

def test_wait_for_model_ready_success(model_client):
    """Test waiting for model to be ready"""
    with patch.object(model_client, 'get_model_status') as mock_status:
        # Mock status changes from processing to complete
        mock_status.side_effect = [
            {"status": "processing"},
            {"status": "processing"},
            {"status": "complete"}
        ]

        with patch('time.sleep'):  # Avoid actual sleeping in tests
            result = model_client.wait_for_model_ready(
                "owner",
                "model",
                check_interval=1
            )

        assert result["status"] == "complete"
        assert mock_status.call_count == 3

def test_wait_for_model_ready_error(model_client):
    """Test waiting for model that ends in error"""
    with patch.object(model_client, 'get_model_status') as mock_status:
        mock_status.return_value = {
            "status": "error",
            "errorMessage": "Test error"
        }

        with patch('time.sleep'):
            with pytest.raises(RuntimeError) as exc_info:
                model_client.wait_for_model_ready("owner", "model")

            assert "Test error" in str(exc_info.value)

def test_wait_for_model_ready_timeout(model_client):
    """Test timeout while waiting for model"""
    with patch.object(model_client, 'get_model_status') as mock_status:
        mock_status.return_value = {"status": "processing"}

        with patch('time.sleep'):
            with pytest.raises(TimeoutError):
                model_client.wait_for_model_ready(
                    "owner",
                    "model",
                    timeout=1,
                    check_interval=0.1
                )

def test_model_metadata_to_dict(sample_metadata):
    """Test ModelMetadata to_dict method"""
    metadata_dict = sample_metadata.to_dict()

    assert metadata_dict['name'] == "test-model"
    assert metadata_dict['framework'] == "PyTorch"
    assert metadata_dict['task_ids'] == ["computer-vision", "image-classification"]
    assert metadata_dict['training_params'] == {"epochs": 100, "batch_size": 32}
