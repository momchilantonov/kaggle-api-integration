import pytest
from unittest.mock import patch, Mock, mock_open
import os
from pathlib import Path
import requests

from src.api.kaggle_client import KaggleAPIClient
from config.settings import API_ENDPOINTS

# Test constants
TEST_CREDENTIALS = {
    'KAGGLE_USERNAME': 'test_user',
    'KAGGLE_KEY': 'test_key'
}

@pytest.fixture
def mock_env():
    """Fixture to mock environment variables"""
    with patch.dict(os.environ, TEST_CREDENTIALS):
        yield

@pytest.fixture
def client(mock_env):
    """Fixture to create a KaggleAPIClient instance"""
    return KaggleAPIClient()

@pytest.fixture
def mock_response():
    """Fixture to create a mock response"""
    response = Mock(spec=requests.Response)
    response.raise_for_status = Mock()
    return response

def test_client_initialization(mock_env):
    """Test client initialization with credentials"""
    client = KaggleAPIClient()
    assert client.auth == (TEST_CREDENTIALS['KAGGLE_USERNAME'], TEST_CREDENTIALS['KAGGLE_KEY'])

def test_client_initialization_no_credentials():
    """Test client initialization with missing credentials"""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(EnvironmentError):
            KaggleAPIClient()

def test_make_request_invalid_endpoint(client):
    """Test making request with invalid endpoint"""
    with pytest.raises(ValueError) as exc_info:
        client._make_request('GET', 'invalid_endpoint')
    assert "Unknown endpoint" in str(exc_info.value)

def test_make_request_retry_success(client, mock_response):
    """Test request retry logic with eventual success"""
    mock_response.raise_for_status.side_effect = [
        requests.exceptions.RequestException,
        requests.exceptions.RequestException,
        None
    ]

    with patch.object(client.session, 'request', return_value=mock_response) as mock_request:
        with patch('time.sleep'):  # Avoid actual sleeping in tests
            response = client._make_request('GET', 'datasets_list')
            assert response == mock_response
            assert mock_request.call_count == 3

def test_make_request_retry_failure(client, mock_response):
    """Test request retry exhaustion"""
    mock_response.raise_for_status.side_effect = requests.exceptions.RequestException

    with patch.object(client.session, 'request', return_value=mock_response):
        with patch('time.sleep'):
            with pytest.raises(requests.exceptions.RequestException):
                client._make_request('GET', 'datasets_list')

def test_http_methods(client, mock_response):
    """Test all HTTP method wrapper functions"""
    methods = {
        'get': client.get,
        'post': client.post,
        'put': client.put,
        'delete': client.delete
    }

    with patch.object(client.session, 'request', return_value=mock_response) as mock_request:
        for method_name, method_func in methods.items():
            response = method_func('datasets_list', params={'param': 'value'})
            assert response == mock_response
            mock_request.assert_called_with(
                method=method_name.upper(),
                url=API_ENDPOINTS['datasets_list'],
                params={'param': 'value'},
                data=None,
                files=None,
                json=None,
                stream=False,
                timeout=30
            )

def test_download_file(client, tmp_path):
    """Test file download functionality"""
    mock_response = Mock()
    mock_response.headers = {'content-length': '100'}
    mock_response.iter_content.return_value = [b'chunk1', b'chunk2']

    filename = 'test.csv'

    with patch('builtins.open', mock_open()) as mock_file:
        result = client.download_file(mock_response, tmp_path, filename)

        assert result == tmp_path / filename
        mock_file.assert_called_once_with(tmp_path / filename, 'wb')
        handle = mock_file()
        assert handle.write.call_count == 2

def test_download_file_no_content_length(client, tmp_path):
    """Test file download without content length header"""
    mock_response = Mock()
    mock_response.headers = {}
    mock_response.content = b'content'

    filename = 'test.csv'

    with patch('builtins.open', mock_open()) as mock_file:
        result = client.download_file(mock_response, tmp_path, filename)

        assert result == tmp_path / filename
        mock_file.assert_called_once_with(tmp_path / filename, 'wb')
        handle = mock_file()
        handle.write.assert_called_once_with(b'content')

def test_upload_file(client, mock_response, tmp_path):
    """Test file upload functionality"""
    file_path = tmp_path / 'test.csv'
    file_path.write_text('test data')
    mock_response.json.return_value = {'success': True}

    with patch.object(client.session, 'request', return_value=mock_response) as mock_request:
        result = client.upload_file('dataset_upload_file', file_path, {'param': 'value'})

        assert result == {'success': True}
        assert mock_request.call_count == 1
        assert mock_request.call_args[1]['files']['file'][0] == 'test.csv'

def test_upload_file_not_found(client):
    """Test upload with non-existent file"""
    with pytest.raises(FileNotFoundError):
        client.upload_file('dataset_upload_file', 'nonexistent.csv')

def test_context_manager(mock_env):
    """Test client as context manager"""
    with patch('requests.Session.close') as mock_close:
        with KaggleAPIClient() as client:
            assert isinstance(client, KaggleAPIClient)
        mock_close.assert_called_once()
