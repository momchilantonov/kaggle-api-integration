import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import hashlib

from src.api.files import FileClient
from src.api.kaggle_client import KaggleAPIClient

@pytest.fixture
def mock_kaggle_client():
    """Fixture to create a mock KaggleAPIClient"""
    return Mock(spec=KaggleAPIClient)

@pytest.fixture
def file_client(mock_kaggle_client):
    """Fixture to create a FileClient with mock KaggleAPIClient"""
    return FileClient(mock_kaggle_client)

def test_get_file(file_client, mock_kaggle_client, tmp_path):
    """Test getting a file"""
    mock_response = Mock()
    mock_kaggle_client.get.return_value = mock_response
    mock_kaggle_client.download_file.return_value = tmp_path / "test.csv"

    result = file_client.get_file(
        "owner",
        "dataset",
        "test.csv",
        tmp_path
    )

    mock_kaggle_client.get.assert_called_once_with(
        'files_get',
        params={
            'datasetOwner': 'owner',
            'datasetName': 'dataset',
            'fileName': 'test.csv'
        },
        stream=True
    )
    assert result == tmp_path / "test.csv"

def test_get_file_existing(file_client, mock_kaggle_client, tmp_path):
    """Test getting a file that already exists"""
    file_path = tmp_path / "test.csv"
    file_path.write_text("existing content")

    result = file_client.get_file(
        "owner",
        "dataset",
        "test.csv",
        tmp_path,
        force=False
    )

    mock_kaggle_client.get.assert_not_called()
    assert result == file_path

def test_get_file_force(file_client, mock_kaggle_client, tmp_path):
    """Test force downloading a file that already exists"""
    file_path = tmp_path / "test.csv"
    file_path.write_text("existing content")

    mock_response = Mock()
    mock_kaggle_client.get.return_value = mock_response
    mock_kaggle_client.download_file.return_value = file_path

    result = file_client.get_file(
        "owner",
        "dataset",
        "test.csv",
        tmp_path,
        force=True
    )

    mock_kaggle_client.get.assert_called_once()
    assert result == file_path

def test_upload_file(file_client, mock_kaggle_client, tmp_path):
    """Test uploading a file"""
    file_path = tmp_path / "test.csv"
    file_path.write_text("test content")

    mock_kaggle_client.upload_file.return_value = {"success": True}

    result = file_client.upload_file(
        "owner",
        "dataset",
        file_path,
        "data/test.csv"
    )

    mock_kaggle_client.upload_file.assert_called_once_with(
        'files_upload',
        file_path,
        {
            'datasetOwner': 'owner',
            'datasetName': 'dataset',
            'path': 'data/test.csv'
        }
    )
    assert result == {"success": True}

def test_upload_file_not_found(file_client):
    """Test uploading a non-existent file"""
    with pytest.raises(FileNotFoundError):
        file_client.upload_file(
            "owner",
            "dataset",
            "nonexistent.csv"
        )

def test_delete_file(file_client, mock_kaggle_client):
    """Test deleting a file"""
    mock_response = Mock()
    mock_response.json.return_value = {"success": True}
    mock_kaggle_client.delete.return_value = mock_response

    result = file_client.delete_file(
        "owner",
        "dataset",
        "test.csv"
    )

    mock_kaggle_client.delete.assert_called_once_with(
        'files_delete',
        params={
            'datasetOwner': 'owner',
            'datasetName': 'dataset',
            'fileName': 'test.csv'
        }
    )
    assert result == {"success": True}

def test_verify_file_hash(file_client, tmp_path):
    """Test file hash verification"""
    file_path = tmp_path / "test.txt"
    content = b"test content"
    file_path.write_bytes(content)

    # Calculate expected hash
    sha256_hash = hashlib.sha256()
    sha256_hash.update(content)
    expected_hash = sha256_hash.hexdigest()

    # Test with correct hash
    assert file_client.verify_file_hash(file_path, expected_hash) is True

    # Test with incorrect hash
    assert file_client.verify_file_hash(file_path, "wrong_hash") is False

def test_verify_file_hash_not_found(file_client, tmp_path):
    """Test hash verification with non-existent file"""
    with pytest.raises(FileNotFoundError):
        file_client.verify_file_hash(
            tmp_path / "nonexistent.txt",
            "some_hash"
        )

def test_get_file_size(file_client, tmp_path):
    """Test getting file size"""
    file_path = tmp_path / "test.txt"
    content = b"test content"
    file_path.write_bytes(content)

    size = file_client.get_file_size(file_path)
    assert size == len(content)

def test_get_file_size_not_found(file_client, tmp_path):
    """Test getting size of non-existent file"""
    with pytest.raises(FileNotFoundError):
        file_client.get_file_size(tmp_path / "nonexistent.txt")
