import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
import json
import zipfile

from src.api.datasets import DatasetClient, DatasetMetadata
from src.api.kaggle_client import KaggleAPIClient

@pytest.fixture
def mock_kaggle_client():
    """Fixture to create a mock KaggleAPIClient"""
    client = Mock(spec=KaggleAPIClient)
    client.credentials = {'username': 'test_user', 'key': 'test_key'}
    return client

@pytest.fixture
def dataset_client(mock_kaggle_client):
    """Fixture to create a DatasetClient with mock KaggleAPIClient"""
    return DatasetClient(mock_kaggle_client)

@pytest.fixture
def sample_metadata():
    """Fixture to create sample dataset metadata"""
    return DatasetMetadata(
        title="Test Dataset",
        slug="test-dataset",
        description="Test description",
        licenses=[{"name": "CC0-1.0"}],
        keywords=["test", "data"],
        collaborators=["user1", "user2"]
    )

def test_list_datasets(dataset_client, mock_kaggle_client):
    """Test listing datasets with various filters"""
    mock_response = Mock()
    mock_response.json.return_value = [
        {"ref": "dataset1"},
        {"ref": "dataset2"}
    ]
    mock_kaggle_client.get.return_value = mock_response

    datasets = dataset_client.list_datasets(
        search="test",
        tags=["tag1", "tag2"],
        page=2,
        page_size=10
    )

    mock_kaggle_client.get.assert_called_once()
    call_params = mock_kaggle_client.get.call_args[1]['params']
    assert call_params['search'] == "test"
    assert call_params['tags'] == "tag1,tag2"
    assert call_params['page'] == 2
    assert call_params['pageSize'] == 10
    assert len(datasets) == 2

def test_download_dataset_file(dataset_client, mock_kaggle_client, tmp_path):
    """Test downloading a specific file from a dataset"""
    mock_response = Mock()
    mock_kaggle_client.get.return_value = mock_response
    mock_kaggle_client.download_file.return_value = tmp_path / "test.csv"

    result = dataset_client.download_dataset(
        "owner",
        "dataset",
        "test.csv",
        tmp_path
    )

    mock_kaggle_client.get.assert_called_once_with(
        'dataset_download',
        params={'datasetSlug': 'owner/dataset', 'fileName': 'test.csv'},
        stream=True
    )
    assert result == tmp_path / "test.csv"

def test_download_dataset_with_unzip(dataset_client, mock_kaggle_client, tmp_path):
    """Test downloading and unzipping a complete dataset"""
    mock_response = Mock()
    mock_kaggle_client.get.return_value = mock_response
    zip_path = tmp_path / "dataset.zip"
    mock_kaggle_client.download_file.return_value = zip_path

    # Create a test zip file
    with zipfile.ZipFile(zip_path, 'w') as test_zip:
        test_zip.writestr('test.csv', 'test data')

    result = dataset_client.download_dataset(
        "owner",
        "dataset",
        path=tmp_path,
        unzip=True
    )

    assert not zip_path.exists()  # Zip should be deleted
    assert (tmp_path / "dataset").exists()  # Extracted folder should exist
    assert result == tmp_path / "dataset"

def test_create_dataset(dataset_client, mock_kaggle_client, tmp_path, sample_metadata):
    """Test creating a new dataset"""
    mock_response = Mock()
    mock_response.json.return_value = {"ref": "new_dataset"}
    mock_kaggle_client.post.return_value = mock_response

    # Create test files
    (tmp_path / "data.csv").write_text("test data")
    (tmp_path / "readme.md").write_text("readme content")

    result = dataset_client.create_dataset(
        tmp_path,
        sample_metadata,
        public=True
    )

    mock_kaggle_client.post.assert_called_once()
    assert mock_kaggle_client.upload_file.call_count == 2  # Two files
    assert result == {"ref": "new_dataset"}

def test_create_dataset_invalid_path(dataset_client, mock_kaggle_client, tmp_path, sample_metadata):
    """Test creating a dataset with invalid path"""
    invalid_path = tmp_path / "nonexistent"

    with pytest.raises(NotADirectoryError):
        dataset_client.create_dataset(invalid_path, sample_metadata)

def test_create_version(dataset_client, mock_kaggle_client, tmp_path):
    """Test creating a new version of a dataset"""
    mock_response = Mock()
    mock_response.json.return_value = {"ref": "new_version"}
    mock_kaggle_client.post.return_value = mock_response

    # Create test files
    (tmp_path / "updated_data.csv").write_text("new data")

    result = dataset_client.create_version(
        tmp_path,
        "Updated data for 2024",
        delete_old_versions=True
    )

    mock_kaggle_client.post.assert_called_once_with(
        'dataset_create_version',
        json={
            'versionNotes': "Updated data for 2024",
            'deleteOldVersions': True
        }
    )
    assert result == {"ref": "new_version"}

def test_create_version_invalid_path(dataset_client, mock_kaggle_client, tmp_path):
    """Test creating a version with invalid path"""
    invalid_path = tmp_path / "nonexistent"

    with pytest.raises(NotADirectoryError):
        dataset_client.create_version(invalid_path, "Test version")

def test_update_metadata(dataset_client, mock_kaggle_client, sample_metadata):
    """Test updating dataset metadata"""
    mock_response = Mock()
    mock_response.json.return_value = {"success": True}
    mock_kaggle_client.post.return_value = mock_response

    result = dataset_client.update_metadata(
        "owner",
        "dataset",
        sample_metadata
    )

    mock_kaggle_client.post.assert_called_once()
    assert result == {"success": True}

def test_metadata_to_dict(sample_metadata):
    """Test DatasetMetadata to_dict method"""
    metadata_dict = sample_metadata.to_dict()

    assert metadata_dict['title'] == "Test Dataset"
    assert metadata_dict['slug'] == "test-dataset"
    assert metadata_dict['description'] == "Test description"
    assert metadata_dict['licenses'] == [{"name": "CC0-1.0"}]
    assert metadata_dict['keywords'] == ["test", "data"]
    assert metadata_dict['collaborators'] == ["user1", "user2"]

def test_delete_dataset(dataset_client, mock_kaggle_client):
    """Test deleting a dataset"""
    mock_response = Mock()
    mock_response.json.return_value = {"success": True}
    mock_kaggle_client.delete.return_value = mock_response

    result = dataset_client.delete_dataset("owner", "dataset")

    mock_kaggle_client.delete.assert_called_once_with(
        'dataset_delete',
        params={
            'ownerSlug': 'owner',
            'datasetSlug': 'dataset'
        }
    )
    assert result == {"success": True}

def test_metadata_validation():
    """Test DatasetMetadata validation"""
    # Test required fields
    with pytest.raises(TypeError):
        DatasetMetadata()

    # Test valid metadata creation
    metadata = DatasetMetadata(
        title="Test",
        slug="test",
        description="Test description",
        licenses=[{"name": "CC0-1.0"}],
        keywords=["test"]
    )
    assert metadata.title == "Test"
    assert metadata.collaborators is None  # Optional field
