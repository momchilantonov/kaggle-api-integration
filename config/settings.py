import os
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
LOGS_DIR = BASE_DIR / 'logs'

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

def get_kaggle_credentials():
    """Get Kaggle credentials from environment variables"""
    return {
        'username': os.getenv('KAGGLE_USERNAME'),
        'key': os.getenv('KAGGLE_KEY')
    }

# API Constants
API_BASE_URL = "https://www.kaggle.com/api/v1"
API_ENDPOINTS = {
    # Datasets endpoints
    'datasets_list': f"{API_BASE_URL}/datasets/list",
    'dataset_download': f"{API_BASE_URL}/datasets/download",
    'dataset_create': f"{API_BASE_URL}/datasets/create",
    'dataset_create_version': f"{API_BASE_URL}/datasets/create/version",
    'dataset_update': f"{API_BASE_URL}/datasets/update",
    'dataset_delete': f"{API_BASE_URL}/datasets/delete",
    'dataset_upload_file': f"{API_BASE_URL}/datasets/upload/file",
    'dataset_metadata': f"{API_BASE_URL}/datasets/metadata",

    # Competitions endpoints
    'competitions_list': f"{API_BASE_URL}/competitions/list",
    'competition_details': f"{API_BASE_URL}/competitions/get",
    'competition_download': f"{API_BASE_URL}/competitions/download",
    'competition_submissions': f"{API_BASE_URL}/competitions/submissions",
    'competition_submit': f"{API_BASE_URL}/competitions/submit",
    'competition_submission_status': f"{API_BASE_URL}/competitions/submissions/status",
    'competition_leaderboard_download': f"{API_BASE_URL}/competitions/leaderboard/download",

    # Kernels (Notebooks) endpoints
    'kernels_list': f"{API_BASE_URL}/kernels/list",
    'kernel_push': f"{API_BASE_URL}/kernels/push",
    'kernel_pull': f"{API_BASE_URL}/kernels/pull",
    'kernel_status': f"{API_BASE_URL}/kernels/status",
    'kernel_output': f"{API_BASE_URL}/kernels/output",
    'kernel_versions': f"{API_BASE_URL}/kernels/list/versions",

    # Models endpoints
    'models_list': f"{API_BASE_URL}/models/list",
    'model_push': f"{API_BASE_URL}/models/push",
    'model_pull': f"{API_BASE_URL}/models/pull",
    'model_status': f"{API_BASE_URL}/models/status",
    'model_initiate': f"{API_BASE_URL}/models/initiate",
    'model_upload': f"{API_BASE_URL}/models/upload",

    # Files endpoints
    'files_get': f"{API_BASE_URL}/files/get",
    'files_upload': f"{API_BASE_URL}/files/upload",
    'files_delete': f"{API_BASE_URL}/files/delete"
}

def setup_logger(name: str, log_file: str, level=logging.INFO):
    """Set up a logger with both file and console handlers"""
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # File handler
    file_handler = RotatingFileHandler(
        LOGS_DIR / log_file,
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Logger setup
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# API Request Settings
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

# Data Processing Settings
CHUNK_SIZE = 8192  # bytes for file downloads
MAX_DOWNLOAD_RETRIES = 3

def validate_environment():
    """Validate that all required environment variables are set"""
    credentials = get_kaggle_credentials()
    missing_vars = [
        key for key, value in credentials.items()
        if value is None
    ]

    if missing_vars:
        raise EnvironmentError(
            f"Missing required environment variables: KAGGLE_"
            f"{', KAGGLE_'.join(missing_vars).upper()}"
        )
    return credentials
