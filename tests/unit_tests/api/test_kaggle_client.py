import pytest
import requests
from pathlib import Path
from unittest.mock import Mock, patch

from src.api.kaggle_client import KaggleAPIClient
from src.utils.error_handlers import AuthenticationError, KaggleAPIError
from src.utils.request_manager import RequestManager

@pytest.fixture
def mock_request_manager():
    return Mock(spec=RequestManager)

def test_client_initialization():
    """Test basic client initialization"""
    client = KaggleAPIClient({'username': 'test_user', 'key': 'test_key'})
    assert client.credentials['username'] == 'test_user'
    assert client.credentials['key'] == 'test_key'

def test_client_initialization_without_credentials():
    """Test client initialization without credentials"""
    with patch('src.utils.auth_validator.AuthValidator.validate_credentials') as mock_validate:
        mock_validate.side_effect = AuthenticationError("No credentials provided")
        with pytest.raises(AuthenticationError):
            KaggleAPIClient(credentials=None)

def test_client_with_invalid_credentials():
    """Test client with invalid credentials"""
    with patch('src.utils.auth_validator.AuthValidator.validate_credentials') as mock_validate:
        mock_validate.side_effect = AuthenticationError("Invalid credentials")
        with pytest.raises(AuthenticationError):
            KaggleAPIClient(credentials={'username': '', 'key': ''})

def test_http_methods(mock_request_manager):
    """Test all HTTP methods"""
    client = KaggleAPIClient({'username': 'test_user', 'key': 'test_key'})
    client.request_manager = mock_request_manager
    endpoint = 'test/endpoint'

    for method in ['get', 'post', 'put', 'delete']:
        mock_response = Mock()
        mock_response.status_code = 200

        method_mock = getattr(mock_request_manager, method)
        method_mock.return_value = mock_response

        response = getattr(client, method)(endpoint)
        method_mock.assert_called_once_with(endpoint)
        assert response == mock_response
        method_mock.reset_mock()

def test_download_file(mock_request_manager, tmp_path):
    """Test file download functionality"""
    client = KaggleAPIClient({'username': 'test_user', 'key': 'test_key'})
    client.request_manager = mock_request_manager

    url = 'https://example.com/test.csv'
    file_path = tmp_path / 'test.csv'

    mock_request_manager.download_file.return_value = file_path
    result = client.download_file(url, file_path)

    mock_request_manager.download_file.assert_called_once_with(url, str(file_path))
    assert result == file_path

def test_context_manager():
    """Test client context manager functionality"""
    credentials = {'username': 'test_user', 'key': 'test_key'}
    client = KaggleAPIClient(credentials)

    # Mock the request_manager's close method
    client.request_manager.close = Mock()

    # Use the context manager
    with client:
        pass

    # Verify close was called
    client.request_manager.close.assert_called_once()

def test_api_error_handling(mock_request_manager):
    """Test API error handling"""
    client = KaggleAPIClient({'username': 'test_user', 'key': 'test_key'})
    client.request_manager = mock_request_manager

    mock_request_manager.get.side_effect = requests.exceptions.HTTPError("API Error")
    with pytest.raises(KaggleAPIError):
        client.get('test/endpoint')

def test_rate_limit_handling(mock_request_manager):
    """Test rate limit handling"""
    client = KaggleAPIClient({'username': 'test_user', 'key': 'test_key'})
    client.request_manager = mock_request_manager

    mock_response = Mock()
    mock_response.headers = {'X-RateLimit-Remaining': '5'}
    mock_request_manager.get.return_value = mock_response

    response = client.get('test/endpoint')
    assert response == mock_response

def test_authentication_validation():
    """Test authentication validation"""
    with patch('src.utils.auth_validator.AuthValidator.validate_credentials') as mock_validate:
        mock_validate.side_effect = AuthenticationError("Invalid credentials")
        with pytest.raises(AuthenticationError):
            KaggleAPIClient(credentials=None)

def test_client_base_url():
    """Test client base URL configuration"""
    client = KaggleAPIClient({'username': 'test_user', 'key': 'test_key'})
    assert client.API_BASE_URL == "https://www.kaggle.com/api/v1"

def test_custom_headers(mock_request_manager):
    """Test custom headers handling"""
    client = KaggleAPIClient({'username': 'test_user', 'key': 'test_key'})
    client.request_manager = mock_request_manager

    custom_headers = {'Custom-Header': 'test-value'}
    client.get('test/endpoint', headers=custom_headers)

    mock_request_manager.get.assert_called_once()
    call_kwargs = mock_request_manager.get.call_args[1]
    assert 'headers' in call_kwargs
    assert call_kwargs['headers'] == custom_headers
