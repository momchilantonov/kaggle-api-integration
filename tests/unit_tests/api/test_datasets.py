import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import requests
import click
import json

from src.api.datasets import DatasetClient, DatasetMetadata
from src.utils.error_handlers import KaggleAPIError

@pytest.fixture
def dataset_client(mock_kaggle_client):
    return DatasetClient(mock_kaggle_client)

@pytest.fixture
def sample_dataset_metadata():
    return DatasetMetadata(
        title="Test Dataset",
        slug="test-dataset",
        description="Test description",
        licenses=[{"name": "CC0-1.0"}],
        keywords=["test", "sample"]
    )

def test_dataset_client_initialization(dataset_client):
    """Test dataset client initialization"""
    assert hasattr(dataset_client, 'client')

def test_list_datasets(dataset_client):
    """Test listing datasets"""
    mock_response = Mock()
    mock_response.json.return_value = [
        {"title": "Dataset 1", "owner": "user1"},
        {"title": "Dataset 2", "owner": "user2"}
    ]

    dataset_client.client.get.return_value = mock_response

    result = dataset_client.list_datasets(search="test", user="testuser")

    dataset_client.client.get.assert_called_once_with(
        'datasets/list',
        params={'search': 'test', 'user': 'testuser', 'page': 1}
    )
    assert len(result) == 2
    assert result[0]["title"] == "Dataset 1"

def test_download_dataset(dataset_client, tmp_path):
    """Test dataset download"""
    # Mock the response for the download request
    mock_response = Mock()
    mock_response.url = "http://example.com/dataset.zip"
    dataset_client.client.get.return_value = mock_response

    # Mock the file download
    download_path = tmp_path / "dataset.zip"
    dataset_client.client.download_file.return_value = download_path

    result = dataset_client.download_dataset(
        owner_slug="owner",
        dataset_slug="dataset",
        path=tmp_path,
        unzip=False
    )

    # Verify the API call
    dataset_client.client.get.assert_called_once_with(
        'datasets/download',
        params={'datasetSlug': 'owner/dataset'},
        stream=True
    )

    assert result == download_path

def test_create_dataset(dataset_client, sample_dataset_metadata, tmp_path):
    """Test dataset creation"""
    # Create test folder with some files
    folder_path = tmp_path / "test_dataset"
    folder_path.mkdir()
    (folder_path / "data.csv").touch()

    # Mock the creation response
    mock_response = Mock()
    mock_response.json.return_value = {"ref": "created/dataset"}
    dataset_client.client.post.return_value = mock_response

    # Mock the _upload_files method to avoid actual file upload
    with patch.object(dataset_client, '_upload_files') as mock_upload:
        result = dataset_client.create_dataset(
            folder_path,
            sample_dataset_metadata,
            public=True
        )

        # Verify only one API call to create dataset
        dataset_client.client.post.assert_called_once_with(
            'datasets/create',
            json={
                **sample_dataset_metadata.to_dict(),
                'isPublic': True
            }
        )

        # Verify files upload was called
        mock_upload.assert_called_once_with(folder_path)
        assert result == {"ref": "created/dataset"}

def test_invalid_dataset_download(dataset_client):
    """Test downloading non-existent dataset"""
    # Mock the API error response
    error_response = Mock()
    error_response.status_code = 404

    # Use HTTPError instead of KaggleAPIError
    dataset_client.client.get.side_effect = requests.exceptions.HTTPError(
        "Dataset not found",
        response=error_response
    )

    with pytest.raises(requests.exceptions.HTTPError):
        dataset_client.download_dataset("invalid", "dataset")

def test_empty_folder_dataset_creation(dataset_client, sample_dataset_metadata, tmp_path):
    """Test dataset creation with empty folder"""
    empty_folder = tmp_path / "empty_dataset"
    empty_folder.mkdir()

    # Mock the file check in the client
    def no_files(*args, **kwargs):
        return []

    empty_folder.glob = Mock(return_value=[])

    with pytest.raises(FileNotFoundError, match="No files found"):
        dataset_client.create_dataset(empty_folder, sample_dataset_metadata)

def test_dataset_with_invalid_metadata(dataset_client, tmp_path):
    """Test dataset creation with invalid metadata"""
    folder_path = tmp_path / "test_dataset"
    folder_path.mkdir()
    (folder_path / "data.csv").touch()

    # Create invalid metadata
    invalid_metadata = DatasetMetadata(
        title="",  # Empty title
        slug="",   # Empty slug
        description="",  # Empty description
        licenses=[],  # Empty licenses
        keywords=[]
    )

    # Mock validation check in create_dataset
    with patch.object(dataset_client, '_validate_metadata') as mock_validate:
        mock_validate.side_effect = ValueError("Invalid metadata")

        with pytest.raises(ValueError, match="Invalid metadata"):
            dataset_client.create_dataset(folder_path, invalid_metadata)

def test_dataset_upload_files(dataset_client, tmp_path):
    """Test uploading files to dataset"""
    # Create test file
    file_path = tmp_path / "test.csv"
    file_path.touch()

    dataset_client._upload_files(tmp_path)

    # Verify file upload API call
    dataset_client.client.post.assert_called_once()
    assert 'datasets/upload/file' in dataset_client.client.post.call_args[0]

def test_invalid_dataset_download(dataset_client):
    """Test downloading non-existent dataset"""
    # Mock the API error response
    mock_response = Mock()
    mock_response.status_code = 404
    mock_response.raise_for_status = Mock(
        side_effect=requests.exceptions.HTTPError(
            "404 Client Error: Not Found for url: test/url",
            response=mock_response
        )
    )
    dataset_client.client.get.return_value = mock_response

    with pytest.raises(click.exceptions.Abort):
        dataset_client.download_dataset("invalid", "dataset")

def test_download_dataset_with_unzip(dataset_client, tmp_path):
    """Test dataset download with unzip option"""
    # Mock the download response
    mock_response = Mock()
    mock_response.url = "http://example.com/dataset.zip"
    dataset_client.client.get.return_value = mock_response

    # Mock the file download
    zip_path = tmp_path / "dataset.zip"
    dataset_client.client.download_file.return_value = zip_path

    # Create test zip file
    zip_path.touch()

    with patch('zipfile.ZipFile'):
        result = dataset_client.download_dataset(
            owner_slug="owner",
            dataset_slug="dataset",
            path=tmp_path,
            unzip=True
        )

        assert result == tmp_path / "dataset"

def test_metadata_conversion(sample_dataset_metadata):
    """Test dataset metadata conversion to dict"""
    metadata_dict = sample_dataset_metadata.to_dict()

    assert metadata_dict['title'] == "Test Dataset"
    assert metadata_dict['slug'] == "test-dataset"
    assert metadata_dict['description'] == "Test description"
    assert metadata_dict['licenses'] == [{"name": "CC0-1.0"}]
    assert metadata_dict['keywords'] == ["test", "sample"]

def test_empty_folder_dataset_creation(dataset_client, sample_dataset_metadata, tmp_path):
    """Test dataset creation with empty folder"""
    empty_folder = tmp_path / "empty_dataset"
    empty_folder.mkdir()

    # Mock the API response for empty folder case
    mock_response = Mock()
    mock_response.status_code = 400  # Bad Request
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        response=mock_response
    )
    dataset_client.client.post.side_effect = requests.exceptions.HTTPError(
        response=mock_response
    )

    with pytest.raises(click.exceptions.Abort):  # Changed to expect Abort
        dataset_client.create_dataset(empty_folder, sample_dataset_metadata)

    # Verify that the post method was called
    dataset_client.client.post.assert_called_once()

def test_dataset_with_invalid_metadata(dataset_client, tmp_path):
    """Test dataset creation with invalid metadata"""
    folder_path = tmp_path / "test_dataset"
    folder_path.mkdir()
    (folder_path / "data.csv").touch()

    # Create invalid metadata
    invalid_metadata = DatasetMetadata(
        title="",  # Empty title
        slug="",   # Empty slug
        description="",  # Empty description
        licenses=[],  # Empty licenses
        keywords=[]
    )

    # Mock the post request to fail with validation error
    mock_response = Mock()
    mock_response.status_code = 422
    mock_response.text = "Invalid metadata"
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        response=mock_response
    )
    dataset_client.client.post.side_effect = requests.exceptions.HTTPError(
        response=mock_response
    )

    with pytest.raises(click.exceptions.Abort):
        dataset_client.create_dataset(folder_path, invalid_metadata)
