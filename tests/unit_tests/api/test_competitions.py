import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import requests

from src.api.competitions import CompetitionClient, SubmissionMetadata
from src.utils.error_handlers import KaggleAPIError
import click

@pytest.fixture
def competition_client(mock_kaggle_client):
    return CompetitionClient(mock_kaggle_client)

@pytest.fixture
def sample_submission_metadata():
    return SubmissionMetadata(
        message="Test submission",
        description="Test description",
        quiet=False
    )

def test_competition_client_initialization(competition_client):
    """Test competition client initialization"""
    assert hasattr(competition_client, 'client')

def test_list_competitions(competition_client):
    """Test listing competitions"""
    mock_response = Mock()
    mock_response.json.return_value = [
        {"title": "Competition 1", "deadline": "2024-12-31"},
        {"title": "Competition 2", "deadline": "2024-12-31"}
    ]
    competition_client.client.get.return_value = mock_response

    result = competition_client.list_competitions(search="test", category="featured")

    competition_client.client.get.assert_called_once_with(
        'competitions/list',
        params={'search': 'test', 'category': 'featured', 'page': 1}
    )
    assert len(result) == 2
    assert result[0]["title"] == "Competition 1"

def test_get_competition_details(competition_client):
    """Test getting competition details"""
    mock_response = Mock()
    mock_response.json.return_value = {
        "title": "Test Competition",
        "deadline": "2024-12-31",
        "reward": "$10,000"
    }
    competition_client.client.get.return_value = mock_response

    result = competition_client.get_competition_details("test-competition")

    competition_client.client.get.assert_called_once_with(
        'competitions/details',
        params={'id': 'test-competition'}
    )
    assert result["title"] == "Test Competition"

def test_download_competition_files(competition_client, tmp_path):
    """Test competition files download"""
    mock_response = Mock()
    mock_response.url = "http://example.com/competition.zip"
    competition_client.client.get.return_value = mock_response

    download_path = tmp_path / "competition.zip"
    competition_client.client.download_file.return_value = download_path

    result = competition_client.download_competition_files(
        competition="test-competition",
        path=tmp_path,
        file_name="competition.zip"
    )

    competition_client.client.get.assert_called_once_with(
        'competitions/download',
        params={'id': 'test-competition', 'fileName': 'competition.zip'},
        stream=True
    )
    assert result == download_path

def test_submit_to_competition(competition_client, sample_submission_metadata, tmp_path):
    """Test competition submission"""
    # Create test submission file
    submission_file = tmp_path / "submission.csv"
    submission_file.touch()

    mock_response = Mock()
    mock_response.json.return_value = {"id": "submission123", "status": "pending"}
    competition_client.client.post.return_value = mock_response

    result = competition_client.submit_to_competition(
        competition="test-competition",
        file_path=submission_file,
        metadata=sample_submission_metadata
    )

    assert result["id"] == "submission123"
    assert result["status"] == "pending"

def test_get_submission_status(competition_client):
    """Test getting submission status"""
    mock_response = Mock()
    mock_response.json.return_value = {
        "id": "submission123",
        "status": "complete",
        "score": 0.95
    }
    competition_client.client.get.return_value = mock_response

    result = competition_client.get_submission_status(
        competition="test-competition",
        submission_id="submission123"
    )

    assert result["status"] == "complete"
    assert result["score"] == 0.95

def test_competition_not_found(competition_client):
    """Test handling of non-existent competition"""
    mock_response = Mock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        response=mock_response
    )
    competition_client.client.get.side_effect = requests.exceptions.HTTPError(
        response=mock_response
    )

    with pytest.raises(click.exceptions.Abort):
        competition_client.get_competition_details("nonexistent-competition")

def test_invalid_submission_file(competition_client, sample_submission_metadata):
    """Test submission with non-existent file"""
    # Mock error handling behavior
    with pytest.raises(click.exceptions.Abort):
        competition_client.submit_to_competition(
            competition="test-competition",
            file_path=Path("nonexistent.csv"),
            metadata=sample_submission_metadata
        )

    # Verify no API call was made
    competition_client.client.post.assert_not_called()

def test_submission_metadata_conversion(sample_submission_metadata):
    """Test submission metadata conversion to dict"""
    metadata_dict = sample_submission_metadata.to_dict()

    assert metadata_dict['message'] == "Test submission"
    assert metadata_dict['description'] == "Test description"
    assert metadata_dict['quiet'] == False

def test_wait_for_submission_scoring(competition_client):
    """Test waiting for submission scoring"""
    # Mock status responses for multiple calls
    mock_responses = [
        {'status': 'pending'},
        {'status': 'processing'},
        {'status': 'complete', 'score': 0.95}
    ]

    competition_client.get_submission_status = Mock(side_effect=mock_responses)

    with patch('time.sleep') as mock_sleep:  # Mock sleep to speed up test
        result = competition_client.wait_for_scoring(
            competition="test-competition",
            submission_id="submission123",
            timeout=3600,
            check_interval=1
        )

    assert result['status'] == 'complete'
    assert result['score'] == 0.95
