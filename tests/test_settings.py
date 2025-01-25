import pytest
import os
import logging
from pathlib import Path
from unittest.mock import patch
from config.settings import (
    setup_logger,
    validate_environment,
    get_kaggle_credentials,
    BASE_DIR,
    DATA_DIR,
    LOGS_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR
)

# Test constants
TEST_CREDENTIALS = {
    'KAGGLE_USERNAME': 'test_user',
    'KAGGLE_KEY': 'test_key'
}

def test_directory_structure():
    """Test that all required directories exist"""
    directories = [DATA_DIR, LOGS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]
    for directory in directories:
        assert directory.exists(), f"Directory {directory} does not exist"
        assert directory.is_dir(), f"{directory} is not a directory"

def test_base_dir_resolution():
    """Test that BASE_DIR is resolved correctly"""
    assert BASE_DIR.exists(), "BASE_DIR does not exist"
    assert BASE_DIR.is_dir(), "BASE_DIR is not a directory"
    assert (BASE_DIR / 'config').exists(), "config directory not found in BASE_DIR"

@pytest.fixture
def temp_log_dir(tmp_path):
    """Fixture to provide a temporary directory for log files"""
    return tmp_path

def test_get_kaggle_credentials():
    """Test getting Kaggle credentials"""
    with patch.dict(os.environ, TEST_CREDENTIALS):
        credentials = get_kaggle_credentials()
        assert credentials['username'] == 'test_user'
        assert credentials['key'] == 'test_key'

def test_get_kaggle_credentials_missing():
    """Test getting Kaggle credentials when they're missing"""
    with patch.dict(os.environ, {}, clear=True):
        credentials = get_kaggle_credentials()
        assert credentials['username'] is None
        assert credentials['key'] is None

def test_environment_validation_success():
    """Test environment validation with valid credentials"""
    with patch.dict(os.environ, TEST_CREDENTIALS):
        credentials = validate_environment()
        assert credentials['username'] == 'test_user'
        assert credentials['key'] == 'test_key'

def test_environment_validation_missing_username():
    """Test environment validation with missing username"""
    with patch.dict(os.environ, {'KAGGLE_KEY': 'test_key'}, clear=True):
        with pytest.raises(EnvironmentError) as exc_info:
            validate_environment()
        assert 'USERNAME' in str(exc_info.value)

def test_environment_validation_missing_key():
    """Test environment validation with missing key"""
    with patch.dict(os.environ, {'KAGGLE_USERNAME': 'test_user'}, clear=True):
        with pytest.raises(EnvironmentError) as exc_info:
            validate_environment()
        assert 'KEY' in str(exc_info.value)

def test_environment_validation_missing_both():
    """Test environment validation with both variables missing"""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(EnvironmentError) as exc_info:
            validate_environment()
        assert 'USERNAME' in str(exc_info.value)
        assert 'KEY' in str(exc_info.value)

def test_logger_setup(temp_log_dir):
    """Test logger configuration"""
    log_file = "test.log"
    logger = setup_logger("test_logger", log_file)

    # Test logger level
    assert logger.level == logging.INFO

    # Test handlers
    assert len(logger.handlers) == 2, "Logger should have 2 handlers (file and console)"

    # Test handler types
    handlers = logger.handlers
    assert any(isinstance(h, logging.handlers.RotatingFileHandler) for h in handlers)
    assert any(isinstance(h, logging.StreamHandler) for h in handlers)

def test_log_file_rotation(temp_log_dir):
    """Test log file rotation functionality"""
    log_file = temp_log_dir / "rotation_test.log"
    logger = setup_logger("rotation_test", str(log_file))

    # Write enough data to trigger rotation
    large_message = "x" * 1024 * 1024  # 1MB message
    for _ in range(11):  # Write 11MB to trigger rotation
        logger.info(large_message)

    # Check that rotation occurred
    log_files = list(temp_log_dir.glob("rotation_test.log*"))
    assert len(log_files) > 1, "Log rotation did not occur"

def test_log_message_format(temp_log_dir):
    """Test log message formatting"""
    log_file = temp_log_dir / "format_test.log"
    logger = setup_logger("format_test", str(log_file))

    test_message = "Test log message"
    logger.info(test_message)

    with open(log_file, 'r') as f:
        log_content = f.read()

    assert test_message in log_content, "Log message not found in file"
    assert " - format_test - INFO - " in log_content, "Incorrect log format"
