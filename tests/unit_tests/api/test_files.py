import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import requests
import click

from src.api.files import FileClient

@pytest.fixture
def file_client(mock_kaggle_client):
    return FileClient(mock_kaggle_client)

def test_file_client_initialization(file_client):
    """Test file client initialization"""
    assert hasattr(file_client, 'client')

def test_get_file(file_client, tmp_path):
    """Test getting a file"""
    # Mock response
    mock_response = Mock()
    mock_response.url = "http://example.com/test.csv"
    file_client.client.get.return_value = mock_response

    # Set up target path
    file_path = tmp_path / "test.csv"
    file_client.client.download_file.return_value = file_path

    result = file_client.get_file(
        dataset_owner="test-owner",
        dataset_name="test-dataset",
        file_name="test.csv",
        path=tmp_path
    )

    # Verify the correct endpoint was called
    file_client.client.get.assert_called_once_with(
        'files/get',
        params={
            'datasetOwner': 'test-owner',
            'datasetName': 'test-dataset',
            'fileName': 'test.csv'
        },
        stream=True
    )
    assert result == file_path

def test_upload_file(file_client, tmp_path):
    """Test uploading a file"""
    # Create test file
    test_file = tmp_path / "test.csv"
    test_file.touch()

    mock_response = Mock()
    mock_response.json.return_value = {"success": True}
    file_client.client.post.return_value = mock_response

    result = file_client.upload_file(
        dataset_owner="test-owner",
        dataset_name="test-dataset",
        file_path=test_file
    )

    assert result["success"] == True
    file_client.client.post.assert_called_once()

def test_delete_file(file_client):
    """Test deleting a file"""
    mock_response = Mock()
    mock_response.json.return_value = {"success": True}
    file_client.client.delete.return_value = mock_response

    result = file_client.delete_file(
        dataset_owner="test-owner",
        dataset_name="test-dataset",
        file_name="test.csv"
    )

    assert result["success"] == True
    file_client.client.delete.assert_called_once_with(
        'files/delete',
        params={
            'datasetOwner': 'test-owner',
            'datasetName': 'test-dataset',
            'fileName': 'test.csv'
        }
    )

def test_get_nonexistent_file(file_client):
    """Test getting a non-existent file"""
    mock_response = Mock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        response=mock_response
    )
    file_client.client.get.side_effect = requests.exceptions.HTTPError(
        response=mock_response
    )

    with pytest.raises(click.exceptions.Abort):
        file_client.get_file(
            dataset_owner="test-owner",
            dataset_name="test-dataset",
            file_name="nonexistent.csv"
        )

def test_upload_nonexistent_file(file_client):
    """Test uploading a non-existent file"""
    with pytest.raises(click.exceptions.Abort):
        file_client.upload_file(
            dataset_owner="test-owner",
            dataset_name="test-dataset",
            file_path=Path("nonexistent.csv")
        )

    # Verify no API call was made
    file_client.client.post.assert_not_called()

def test_get_file_with_force_option(file_client, tmp_path):
    """Test getting a file with force option"""
    # Create existing file
    existing_file = tmp_path / "test.csv"
    existing_file.touch()

    mock_response = Mock()
    mock_response.url = "http://example.com/test.csv"
    file_client.client.get.return_value = mock_response

    file_client.client.download_file.return_value = existing_file

    result = file_client.get_file(
        dataset_owner="test-owner",
        dataset_name="test-dataset",
        file_name="test.csv",
        path=tmp_path,
        force=True
    )

    assert result == existing_file
    file_client.client.get.assert_called_once()

def test_upload_file_with_target_path(file_client, tmp_path):
    """Test uploading a file with custom target path"""
    test_file = tmp_path / "test.csv"
    test_file.touch()

    mock_response = Mock()
    mock_response.json.return_value = {"success": True}
    file_client.client.post.return_value = mock_response

    target_path = "custom/path/test.csv"
    result = file_client.upload_file(
        dataset_owner="test-owner",
        dataset_name="test-dataset",
        file_path=test_file,
        target_path=target_path
    )

    assert result["success"] == True
    assert file_client.client.post.call_args[1]['data']['path'] == target_path

def test_delete_file_error(file_client):
    """Test error when deleting a file"""
    mock_response = Mock()
    mock_response.status_code = 400
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        response=mock_response
    )
    file_client.client.delete.side_effect = requests.exceptions.HTTPError(
        response=mock_response
    )

    with pytest.raises(click.exceptions.Abort):
        file_client.delete_file(
            dataset_owner="test-owner",
            dataset_name="test-dataset",
            file_name="test.csv"
        )

def test_get_file_existing_no_force(file_client, tmp_path):
    """Test getting a file when it exists and force is False"""
    existing_file = tmp_path / "test.csv"
    existing_file.touch()

    result = file_client.get_file(
        dataset_owner="test-owner",
        dataset_name="test-dataset",
        file_name="test.csv",
        path=tmp_path,
        force=False
    )

    assert result == existing_file
    file_client.client.get.assert_not_called()
