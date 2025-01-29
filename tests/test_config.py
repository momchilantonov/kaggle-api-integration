import pytest
from pathlib import Path
import yaml
import json
import click
from src.utils.config_validator import ConfigValidator
from src.utils.error_handlers import CLIError

@pytest.fixture
def config_validator():
    return ConfigValidator()

@pytest.fixture
def sample_config_dir(tmp_path):
    config_dir = tmp_path / "operational_configs"
    config_dir.mkdir()
    return config_dir

@pytest.fixture
def valid_dataset_config():
    return {
        'frequently_used_datasets': {
            'titanic': {
                'owner': 'kaggle',
                'dataset': 'titanic',
                'files': ['train.csv', 'test.csv'],
                'local_path': 'data/datasets/titanic'
            }
        }
    }

def test_validate_dataset_config(config_validator, sample_config_dir, valid_dataset_config):
    config_path = sample_config_dir / "dataset_configs"
    config_path.mkdir()
    config_file = config_path / "datasets.yaml"

    with open(config_file, 'w') as f:
        yaml.dump(valid_dataset_config, f)

    result = config_validator.validate_config(
        config_file,
        'dataset_configs',
        'datasets'
    )
    assert result == valid_dataset_config

def test_validate_invalid_yaml(config_validator, sample_config_dir):
    config_path = sample_config_dir / "invalid.yaml"
    with open(config_path, 'w') as f:
        f.write("invalid: yaml: content:")

    with pytest.raises(yaml.YAMLError):
        config_validator.validate_config(
            config_path,
            'dataset_configs',
            'datasets'
        )

def test_verify_paths(config_validator):
    test_config = {
        'data_path': 'test/data',
        'backup_path': 'test/backups',
        'nested': {
            'path': 'test/nested/path'
        }
    }

    errors = config_validator.verify_paths(test_config)
    assert isinstance(errors, list)

    # All paths should be created
    for path in ['test/data', 'test/backups', 'test/nested/path']:
        assert Path(path).exists()

def test_validate_all_configs(config_validator, sample_config_dir):
    # Create test config files with proper structure
    dataset_config = sample_config_dir / "dataset_configs"
    dataset_config.mkdir(parents=True)

    # Create a valid datasets.yaml file
    with open(dataset_config / "datasets.yaml", 'w') as f:
        yaml.dump({
            'frequently_used_datasets': {
                'titanic': {
                    'owner': 'kaggle',
                    'dataset': 'titanic',
                    'files': ['train.csv', 'test.csv'],
                    'local_path': 'data/datasets/titanic'
                }
            }
        }, f)

    configs = config_validator.validate_all_configs(sample_config_dir)

    # Check that the configs were loaded correctly
    assert 'dataset_configs' in configs
    assert isinstance(configs['dataset_configs'], dict)
    assert 'datasets' in configs['dataset_configs']
    assert 'frequently_used_datasets' in configs['dataset_configs']['datasets']

def test_invalid_config_schema(config_validator, sample_config_dir):
    invalid_config = {
        'frequently_used_datasets': {
            'titanic': {
                # Missing required 'owner' field
                'dataset': 'titanic',
                'files': ['train.csv']
            }
        }
    }

    config_path = sample_config_dir / "dataset_configs"
    config_path.mkdir()
    config_file = config_path / "datasets.yaml"

    with open(config_file, 'w') as f:
        yaml.dump(invalid_config, f)

    with pytest.raises(Exception):
        config_validator.validate_config(
            config_file,
            'dataset_configs',
            'datasets'
        )

def test_nonexistent_config_file(config_validator):
    with pytest.raises(FileNotFoundError):
        config_validator.validate_config(
            Path("nonexistent.yaml"),
            'dataset_configs',
            'datasets'
        )

def test_path_permission_handling(config_validator, tmp_path):
    # Create parent directory
    test_dir = tmp_path / 'test_dir'
    test_dir.mkdir()

    # Create a file where we want a directory to block path creation
    blocked_path = test_dir / 'blocked'
    blocked_path.write_text('')  # Create as file instead of directory

    test_config = {
        'data_path': str(blocked_path / 'data')  # Try to create directory where file exists
    }

    # Verify paths - should return error because we can't create a directory where a file exists
    errors = config_validator.verify_paths(test_config)

    assert len(errors) > 0
    assert any('cannot create path' in error.lower() for error in errors)
