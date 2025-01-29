from pathlib import Path
import logging
from typing import Union, List
import shutil

logger = logging.getLogger(__name__)

class PathManager:
    """Manages file paths and directory operations for workflows"""

    def __init__(self, base_path: Union[str, Path] = None):
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.required_dirs = {
            'competitions': {
                'data': 'data/competitions',
                'submissions': 'data/competitions/submissions',
                'leaderboards': 'data/competitions/leaderboards'
            },
            'datasets': {
                'raw': 'data/datasets/raw',
                'processed': 'data/datasets/processed',
                'uploads': 'data/datasets/uploads'
            },
            'models': {
                'custom': 'data/models/custom',
                'downloaded': 'data/models/downloaded',
                'checkpoints': 'data/models/checkpoints'
            },
            'kernels': {
                'scripts': 'data/kernels/scripts',
                'outputs': 'data/kernels/outputs'
            },
            'logs': 'logs'
        }

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist"""
        try:
            def create_nested_dirs(struct, parent=None):
                if isinstance(struct, dict):
                    for name, path in struct.items():
                        if isinstance(path, dict):
                            create_nested_dirs(path, parent)
                        else:
                            full_path = self.base_path / path
                            full_path.mkdir(parents=True, exist_ok=True)
                            logger.info(f"Created directory: {full_path}")
                else:
                    full_path = self.base_path / struct
                    full_path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created directory: {full_path}")

            create_nested_dirs(self.required_dirs)
        except Exception as e:
            logger.error(f"Error creating directories: {str(e)}")
            raise

    def get_path(self, category: str, subcategory: str = None) -> Path:
        """Get path for specific category and subcategory"""
        try:
            if subcategory:
                path = self.required_dirs[category][subcategory]
            else:
                path = self.required_dirs[category]
            return self.base_path / path
        except KeyError:
            raise ValueError(f"Invalid category or subcategory: {category}/{subcategory}")

    def clean_directory(self, path: Union[str, Path], pattern: str = "*") -> None:
        """Clean files in directory matching pattern"""
        try:
            path = Path(path)
            for item in path.glob(pattern):
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            logger.info(f"Cleaned directory: {path}")
        except Exception as e:
            logger.error(f"Error cleaning directory {path}: {str(e)}")
            raise

    def create_temp_directory(self) -> Path:
        """Create and return a temporary directory"""
        try:
            temp_dir = self.base_path / "data/temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            return temp_dir
        except Exception as e:
            logger.error(f"Error creating temp directory: {str(e)}")
            raise

    def backup_file(self, file_path: Union[str, Path], backup_dir: str = "backups") -> Path:
        """Create backup of a file"""
        try:
            file_path = Path(file_path)
            backup_path = self.base_path / backup_dir / file_path.name
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Error creating backup: {str(e)}")
            raise
