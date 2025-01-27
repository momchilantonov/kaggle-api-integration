import os
import logging
from typing import Dict, Optional
from pathlib import Path
from .error_handlers import AuthenticationError

logger = logging.getLogger(__name__)

class AuthValidator:
    REQUIRED_ENV_VARS = ['KAGGLE_USERNAME', 'KAGGLE_KEY']
    CONFIG_FILE = '.kaggle/kaggle.json'

    @staticmethod
    def validate_credentials(credentials: Optional[Dict] = None) -> Dict:
        """Validate Kaggle credentials"""
        if credentials and all(credentials.values()):
            return credentials

        # Check environment variables
        env_credentials = {
            'username': os.getenv('KAGGLE_USERNAME'),
            'key': os.getenv('KAGGLE_KEY')
        }
        if all(env_credentials.values()):
            return env_credentials

        # Check config file
        config_credentials = AuthValidator._load_config_file()
        if config_credentials:
            return config_credentials

        raise AuthenticationError("No valid credentials found")

    @staticmethod
    def _load_config_file() -> Optional[Dict]:
        """Load credentials from config file"""
        config_path = Path.home() / AuthValidator.CONFIG_FILE

        if not config_path.exists():
            return None

        try:
            import json
            with open(config_path) as f:
                config = json.load(f)
                return {
                    'username': config.get('username'),
                    'key': config.get('key')
                }
        except Exception as e:
            logger.error(f"Error loading config file: {str(e)}")
            return None

    @staticmethod
    def validate_permissions(config_path: Path) -> None:
        """Validate config file permissions"""
        import stat
        if config_path.exists():
            current_mode = config_path.stat().st_mode
            if current_mode & (stat.S_IRWXG | stat.S_IRWXO):
                logger.warning("Config file has too permissive permissions")
                config_path.chmod(stat.S_IRUSR | stat.S_IWUSR)

    @staticmethod
    def create_config(credentials: Dict) -> None:
        """Create configuration file"""
        config_path = Path.home() / AuthValidator.CONFIG_FILE
        config_path.parent.mkdir(parents=True, exist_ok=True)

        import json
        with open(config_path, 'w') as f:
            json.dump(credentials, f)

        AuthValidator.validate_permissions(config_path)
