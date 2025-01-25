import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
import json
import time

from src.api.competitions import CompetitionClient, SubmissionMetadata
from src.api.kaggle_client import KaggleAPIClient

@pytest.fixture
def mock_kaggle_client():
    """Fixture to create a mock KaggleAPIClient"""
    return Mock(spec=KaggleAPIClient)

@pytest.fixture
def competition_client(mock_kaggle_client):
    """Fixture to create a CompetitionClient with mock KaggleAPIClient"""
    return CompetitionClient(mock_kaggle_client)

@pytest.fixture
def sample_metadata():
    """Fixture to create sample submission metadata"""
    return SubmissionMetadata(
        message="Test submission",
        description="Testing competition submission",
        quiet=False
    )

def test_list_competitions(competition_client, mock_kaggle_client):
    """Test listing competitions with filters"""
    mock_response = Mock()
    mock_response.json.return_value = [
        {"title": "Competition 1", "category": "Featured"},
        {"title": "Competition 2", "category": "Research"}
    ]
    mock_kaggle_client.get.return_value = mock_response

    competitions = competition_client.list_competitions(
        search="test",
        category="Featured",
        group="active",
        page=2
    )

    mock_kaggle_client.get.assert_called_once()
    call_params = mock_kaggle_client.get.call_args[1]['params']
    assert call_params['search'] == "test"
    assert call_params['category'] == "Featured"
    assert call_params['group'] == "active"
    assert call_params['page'] == 2
    assert len(competitions) == 2

def test_get_competition_details(competition_client, mock_kaggle_client):
    """Test getting competition details"""
    mock_response = Mock()
    mock_response.json.return_value = {
        "title": "Test Competition",
        "description": "Test Description",
        "evaluationMetric": "AUC"
    }
    mock_kaggle_client.get.return_value = mock_response

    details = competition_client.get_competition_details("test-competition")

    mock_kaggle_client.get.assert_called_once_with(
        'competition_details',
        params={'id': 'test-competition'}
    )
    assert details['title'] == "Test Competition"
    assert details['evaluationMetric'] == "AUC"

def test_download_competition_files(competition_client, mock_kaggle_client, tmp_path):
    """Test downloading competition files"""
    mock_response = Mock()
    mock_kaggle_client.get.return_value = mock_response
    mock_kaggle_client.download_file.return_value = tmp_path / "test-competition.zip"

    result = competition_client.download_competition_files(
        "test-competition",
        tmp_path
    )

    mock_kaggle_client.get.assert_called_once_with(
        'competition_download',
        params={'id': 'test-competition'},
        stream=True
    )
    assert result == tmp_path / "test-competition.zip"

def test_download_specific_competition_file(competition_client, mock_kaggle_client, tmp_path):
    """Test downloading a specific competition file"""
    mock_response = Mock()
    mock_kaggle_client.get.return_value = mock_response
    mock_kaggle_client.download_file.return_value = tmp_path / "train.csv"

    result = competition_client.download_competition_files(
        "test-competition",
        tmp_path,
        "train.csv"
    )

    mock_kaggle_client.get.assert_called_once_with(
        'competition_download',
        params={
            'id': 'test-competition',
            'fileName': 'train.csv'
        },
        stream=True
    )
    assert result == tmp_path / "train.csv"

def test_submit_to_competition(competition_client, mock_kaggle_client, tmp_path, sample_metadata):
    """Test submitting to a competition"""
    submission_file = tmp_path / "submission.csv"
    submission_file.write_text("id,prediction\n1,0\n2,1")

    mock_response = Mock()
    mock_response.json.return_value = {"id": "submission123", "status": "pending"}
    mock_kaggle_client.post.return_value = mock_response

    result = competition_client.submit_to_competition(
        "test-competition",
        submission_file,
        sample_metadata
    )

    mock_kaggle_client.post.assert_called_once()
    assert result["id"] == "submission123"
    assert result["status"] == "pending"

def test_submit_to_competition_missing_file(competition_client, sample_metadata, tmp_path):
    """Test submitting with missing file"""
    with pytest.raises(FileNotFoundError):
        competition_client.submit_to_competition(
            "test-competition",
            tmp_path / "nonexistent.csv",
            sample_metadata
        )

def test_get_submission_status(competition_client, mock_kaggle_client):
    """Test getting submission status"""
    mock_response = Mock()
    mock_response.json.return_value = {
        "id": "submission123",
        "status": "complete",
        "score": 0.95
    }
    mock_kaggle_client.get.return_value = mock_response

    status = competition_client.get_submission_status(
        "test-competition",
        "submission123"
    )

    mock_kaggle_client.get.assert_called_once_with(
        'competition_submission_status',
        params={
            'id': 'test-competition',
            'submissionId': 'submission123'
        }
    )
    assert status["status"] == "complete"
    assert status["score"] == 0.95

def test_get_competition_submissions(competition_client, mock_kaggle_client):
    """Test getting competition submissions"""
    mock_response = Mock()
    mock_response.json.return_value = [
        {"id": "sub1", "score": 0.95},
        {"id": "sub2", "score": 0.93}
    ]
    mock_kaggle_client.get.return_value = mock_response

    submissions = competition_client.get_competition_submissions(
        "test-competition",
        page=2,
        page_size=10
    )

    mock_kaggle_client.get.assert_called_once_with(
        'competition_submissions',
        params={
            'id': 'test-competition',
            'page': 2,
            'pageSize': 10
        }
    )
    assert len(submissions) == 2
    assert submissions[0]["score"] == 0.95

def test_wait_for_submission_completion_success(competition_client):
    """Test waiting for submission completion - success case"""
    with patch.object(competition_client, 'get_submission_status') as mock_status:
        mock_status.return_value = {"status": "complete", "score": 0.95}

        with patch('time.sleep'):
            result = competition_client.wait_for_submission_completion(
                "test-competition",
                "submission123"
            )

        assert result["status"] == "complete"
        assert result["score"] == 0.95

def test_wait_for_submission_completion_failure(competition_client):
    """Test waiting for submission completion - failure case"""
    with patch.object(competition_client, 'get_submission_status') as mock_status:
        mock_status.return_value = {
            "status": "failed",
            "errorMessage": "Test error"
        }

        with patch('time.sleep'):
            with pytest.raises(RuntimeError) as exc_info:
                competition_client.wait_for_submission_completion(
                    "test-competition",
                    "submission123"
                )

            assert "Test error" in str(exc_info.value)

def test_wait_for_submission_completion_timeout(competition_client):
    """Test timeout while waiting for submission"""
    with patch.object(competition_client, 'get_submission_status') as mock_status:
        mock_status.return_value = {"status": "processing"}

        with patch('time.sleep'):
            with pytest.raises(TimeoutError):
                competition_client.wait_for_submission_completion(
                    "test-competition",
                    "submission123",
                    timeout=1,
                    check_interval=0.1
                )

def test_download_leaderboard(competition_client, mock_kaggle_client, tmp_path):
    """Test downloading competition leaderboard"""
    mock_response = Mock()
    mock_response.content = b"team,score\nteam1,0.95\nteam2,0.93"
    mock_kaggle_client.get.return_value = mock_response

    result = competition_client.download_leaderboard(
        "test-competition",
        tmp_path
    )

    mock_kaggle_client.get.assert_called_once_with(
        'competition_leaderboard_download',
        params={'id': 'test-competition'}
    )

    assert result == tmp_path / "test-competition_leaderboard.csv"
    assert result.exists()
    assert result.read_bytes() == b"team,score\nteam1,0.95\nteam2,0.93"

def test_submission_metadata_to_dict(sample_metadata):
    """Test SubmissionMetadata to_dict method"""
    metadata_dict = sample_metadata.to_dict()

    assert metadata_dict['message'] == "Test submission"
    assert metadata_dict['description'] == "Testing competition submission"
    assert metadata_dict['quiet'] is False
