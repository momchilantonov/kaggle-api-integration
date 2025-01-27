import yaml
from pathlib import Path
from typing import Dict, Any, List
import jsonschema
import logging

logger = logging.getLogger(__name__)

DATASET_CONFIG_SCHEMAS = {
    'datasets': {
        'type': 'object',
        'required': ['frequently_used_datasets'],
        'properties': {
            'frequently_used_datasets': {
                'type': 'object',
                'patternProperties': {
                    '^.*$': {  # Pattern for any dataset name
                        'type': 'object',
                        'required': ['owner', 'dataset', 'files'],
                        'properties': {
                            'owner': {'type': 'string'},
                            'dataset': {'type': 'string'},
                            'files': {
                                'type': 'array',
                                'items': {'type': 'string'}
                            },
                            'local_path': {'type': 'string'}
                        }
                    }
                }
            }
        }
    },
    'download_settings': {
        'type': 'object',
        'required': ['download_preferences'],
        'properties': {
            'download_preferences': {
                'type': 'object',
                'required': ['default_path', 'auto_extract'],
                'properties': {
                    'default_path': {'type': 'string'},
                    'auto_extract': {'type': 'boolean'},
                    'create_backup': {'type': 'boolean'},
                    'file_types': {
                        'type': 'array',
                        'items': {'type': 'string'}
                    },
                    'max_file_size': {'type': 'integer'},
                    'chunk_size': {'type': 'integer'},
                    'verify_hash': {'type': 'boolean'},
                    'retry_settings': {
                        'type': 'object',
                        'properties': {
                            'max_retries': {'type': 'integer'},
                            'delay': {'type': 'integer'},
                            'backoff_factor': {'type': 'number'}
                        }
                    },
                    'parallel_downloads': {
                        'type': 'object',
                        'properties': {
                            'enabled': {'type': 'boolean'},
                            'max_workers': {'type': 'integer'}
                        }
                    },
                    'processing': {
                        'type': 'object',
                        'properties': {
                            'handle_missing': {'type': 'boolean'},
                            'data_types': {
                                'type': 'object',
                                'patternProperties': {
                                    '^.*$': {'type': 'string'}
                                }
                            },
                            'encoding': {'type': 'string'},
                            'compression': {
                                'type': 'object',
                                'properties': {
                                    'enabled': {'type': 'boolean'},
                                    'method': {'type': 'string'},
                                    'threshold': {'type': 'integer'}
                                }
                            }
                        }
                    }
                }
            },
            'storage_settings': {
                'type': 'object',
                'properties': {
                    'base_path': {'type': 'string'},
                    'directory_structure': {
                        'type': 'object',
                        'properties': {
                            'raw': {'type': 'string'},
                            'processed': {'type': 'string'},
                            'temp': {'type': 'string'},
                            'archive': {'type': 'string'}
                        }
                    },
                    'cleanup': {
                        'type': 'object',
                        'properties': {
                            'enabled': {'type': 'boolean'},
                            'temp_file_ttl': {'type': 'integer'},
                            'archive_after_days': {'type': 'integer'}
                        }
                    }
                }
            },
            'monitoring': {
                'type': 'object',
                'properties': {
                    'log_level': {'type': 'string'},
                    'track_downloads': {'type': 'boolean'},
                    'track_processing': {'type': 'boolean'},
                    'alerts': {
                        'type': 'object',
                        'properties': {
                            'size_threshold': {'type': 'integer'},
                            'time_threshold': {'type': 'integer'},
                            'error_notification': {'type': 'boolean'}
                        }
                    }
                }
            }
        }
    }
}

FILE_CONFIG_SCHEMAS = {
    'file_operations': {
        'type': 'object',
        'required': ['file_settings', 'backup_settings', 'file_patterns', 'validation_settings'],
        'properties': {
            'file_settings': {
                'type': 'object',
                'required': ['downloads', 'uploads', 'processing'],
                'properties': {
                    'downloads': {
                        'type': 'object',
                        'required': ['default_path', 'create_directories'],
                        'properties': {
                            'default_path': {'type': 'string'},
                            'create_directories': {'type': 'boolean'},
                            'overwrite_existing': {'type': 'boolean'},
                            'backup_existing': {'type': 'boolean'},
                            'chunk_size': {'type': 'integer'},
                            'max_retries': {'type': 'integer'},
                            'retry_delay': {'type': 'integer'}
                        }
                    },
                    'uploads': {
                        'type': 'object',
                        'required': ['default_path', 'max_file_size'],
                        'properties': {
                            'default_path': {'type': 'string'},
                            'max_file_size': {'type': 'integer'},
                            'allowed_extensions': {
                                'type': 'array',
                                'items': {'type': 'string'}
                            },
                            'manifest_path': {'type': 'string'}
                        }
                    },
                    'processing': {
                        'type': 'object',
                        'required': ['chunk_size', 'encoding'],
                        'properties': {
                            'chunk_size': {'type': 'integer'},
                            'compression_type': {'type': 'string'},
                            'encoding': {'type': 'string'},
                            'temp_directory': {'type': 'string'}
                        }
                    }
                }
            },
            'backup_settings': {
                'type': 'object',
                'required': ['enabled', 'path'],
                'properties': {
                    'enabled': {'type': 'boolean'},
                    'path': {'type': 'string'},
                    'retention_days': {'type': 'integer'},
                    'max_backups': {'type': 'integer'},
                    'compression': {'type': 'boolean'},
                    'include_timestamp': {'type': 'boolean'}
                }
            },
            'file_patterns': {
                'type': 'object',
                'properties': {
                    'data_files': {'type': 'string'},
                    'model_files': {'type': 'string'},
                    'config_files': {'type': 'string'},
                    'archive_files': {'type': 'array', 'items': {'type': 'string'}},
                    'text_files': {'type': 'array', 'items': {'type': 'string'}}
                }
            },
            'validation_settings': {
                'type': 'object',
                'required': ['max_file_size_mb'],
                'properties': {
                    'max_file_size_mb': {'type': 'integer'},
                    'allowed_mime_types': {'type': 'array', 'items': {'type': 'string'}},
                    'validate_encoding': {'type': 'boolean'},
                    'check_file_headers': {'type': 'boolean'}
                }
            }
        }
    },
    'file_paths': {
        'type': 'object',
        'required': ['base_paths'],
        'properties': {
            'base_paths': {
                'type': 'object',
                'required': ['root', 'downloads', 'uploads'],
                'properties': {
                    'root': {'type': 'string'},
                    'downloads': {'type': 'string'},
                    'uploads': {'type': 'string'},
                    'processed': {'type': 'string'},
                    'temp': {'type': 'string'},
                    'backups': {'type': 'string'},
                    'logs': {'type': 'string'}
                }
            },
            'dataset_paths': {
                'type': 'object',
                'patternProperties': {
                    '^.*$': {
                        'type': 'object',
                        'properties': {
                            'train': {'type': 'string'},
                            'test': {'type': 'string'},
                            'submission': {'type': 'string'}
                        }
                    }
                }
            },
            'output_paths': {
                'type': 'object',
                'properties': {
                    'visualizations': {'type': 'string'},
                    'reports': {'type': 'string'},
                    'metrics': {'type': 'string'},
                    'predictions': {'type': 'string'}
                }
            }
        }
    }
}

MODEL_CONFIG_SCHEMAS = {
    'model_params': {
        'type': 'object',
        'required': ['model_settings', 'frameworks', 'model_types', 'resource_limits'],
        'properties': {
            'model_settings': {
                'type': 'object',
                'required': ['default_visibility', 'upload_timeout'],
                'properties': {
                    'default_visibility': {'type': 'boolean'},
                    'upload_timeout': {'type': 'integer'},
                    'processing_timeout': {'type': 'integer'},
                    'version_control': {
                        'type': 'object',
                        'properties': {
                            'max_versions': {'type': 'integer'},
                            'keep_latest': {'type': 'boolean'},
                            'archive_old_versions': {'type': 'boolean'},
                            'version_naming': {'type': 'string'}
                        }
                    }
                }
            },
            'frameworks': {
                'type': 'object',
                'patternProperties': {
                    '^.*$': {  # Pattern for framework names (pytorch, tensorflow, etc.)
                        'type': 'object',
                        'required': ['files', 'required_files'],
                        'properties': {
                            'files': {
                                'type': 'array',
                                'items': {'type': 'string'}
                            },
                            'required_files': {
                                'type': 'array',
                                'items': {'type': 'string'}
                            },
                            'version': {'type': 'string'},
                            'dependencies': {
                                'type': 'array',
                                'items': {'type': 'string'}
                            }
                        }
                    }
                }
            },
            'model_types': {
                'type': 'object',
                'patternProperties': {
                    '^.*$': {  # Pattern for model types
                        'type': 'object',
                        'required': ['task_ids', 'default_framework'],
                        'properties': {
                            'task_ids': {
                                'type': 'array',
                                'items': {'type': 'string'}
                            },
                            'default_framework': {'type': 'string'},
                            'metrics': {
                                'type': 'array',
                                'items': {'type': 'string'}
                            }
                        }
                    }
                }
            },
            'resource_limits': {
                'type': 'object',
                'properties': {
                    'max_model_size': {'type': 'integer'},
                    'max_artifact_size': {'type': 'integer'},
                    'max_total_size': {'type': 'integer'},
                    'memory_limit': {'type': 'string'},
                    'gpu_memory_limit': {'type': 'string'}
                }
            }
        }
    },
    'training_config': {
        'type': 'object',
        'required': ['training_settings', 'data_handling', 'monitoring'],
        'properties': {
            'training_settings': {
                'type': 'object',
                'required': ['default', 'gpu_settings', 'checkpointing'],
                'properties': {
                    'default': {
                        'type': 'object',
                        'properties': {
                            'batch_size': {'type': 'integer'},
                            'epochs': {'type': 'integer'},
                            'learning_rate': {'type': 'number'},
                            'validation_split': {'type': 'number'},
                            'early_stopping': {'type': 'boolean'},
                            'patience': {'type': 'integer'},
                            'optimizer': {'type': 'string'},
                            'save_best_only': {'type': 'boolean'}
                        }
                    },
                    'gpu_settings': {
                        'type': 'object',
                        'properties': {
                            'memory_limit': {'type': 'integer'},
                            'allow_growth': {'type': 'boolean'},
                            'mixed_precision': {'type': 'boolean'},
                            'multi_gpu': {'type': 'boolean'},
                            'gpu_count': {'type': 'integer'}
                        }
                    },
                    'checkpointing': {
                        'type': 'object',
                        'properties': {
                            'enabled': {'type': 'boolean'},
                            'frequency': {'type': 'string'},
                            'max_to_keep': {'type': 'integer'},
                            'save_format': {'type': 'string'},
                            'include_optimizer': {'type': 'boolean'}
                        }
                    }
                }
            },
            'data_handling': {
                'type': 'object',
                'properties': {
                    'augmentation': {
                        'type': 'object',
                        'properties': {
                            'enabled': {'type': 'boolean'},
                            'techniques': {
                                'type': 'array',
                                'items': {'type': 'string'}
                            },
                            'probability': {'type': 'number'}
                        }
                    },
                    'preprocessing': {
                        'type': 'object',
                        'properties': {
                            'normalization': {'type': 'string'},
                            'handle_missing': {'type': 'string'},
                            'categorical_encoding': {'type': 'string'},
                            'text_preprocessing': {
                                'type': 'object',
                                'properties': {
                                    'lowercase': {'type': 'boolean'},
                                    'remove_punctuation': {'type': 'boolean'},
                                    'remove_numbers': {'type': 'boolean'}
                                }
                            }
                        }
                    }
                }
            },
            'monitoring': {
                'type': 'object',
                'properties': {
                    'metrics': {
                        'type': 'array',
                        'items': {'type': 'string'}
                    },
                    'logging': {
                        'type': 'object',
                        'properties': {
                            'log_frequency': {'type': 'integer'},
                            'profile_batch': {'type': 'string'},
                            'log_images': {'type': 'boolean'},
                            'log_graphs': {'type': 'boolean'}
                        }
                    },
                    'visualization': {
                        'type': 'object',
                        'properties': {
                            'enabled': {'type': 'boolean'},
                            'plots': {
                                'type': 'array',
                                'items': {'type': 'string'}
                            },
                            'save_format': {'type': 'string'}
                        }
                    }
                }
            }
        }
    }
}



CONFIG_SCHEMAS = {}

class ConfigValidator:
    @staticmethod
    def validate_config(config_path: Path, schema_type: str, schema_name: str) -> Dict:
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            schema = CONFIG_SCHEMAS[schema_type][schema_name]
            jsonschema.validate(instance=config, schema=schema)
            return config
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {config_path}: {e}")
            raise
        except jsonschema.exceptions.ValidationError as e:
            logger.error(f"Config validation failed for {config_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error validating config {config_path}: {e}")
            raise

    @staticmethod
    def validate_all_configs(config_dir: Path) -> Dict[str, Dict]:
        configs = {}
        for category in config_dir.iterdir():
            if category.is_dir():
                configs[category.name] = {}
                for config_file in category.glob('*.yaml'):
                    try:
                        config_name = config_file.stem
                        configs[category.name][config_name] = ConfigValidator.validate_config(
                            config_file,
                            category.name,
                            config_name
                        )
                    except Exception as e:
                        logger.error(f"Failed to validate {config_file}: {e}")
        return configs

    @staticmethod
    def verify_paths(config: Dict) -> List[str]:
        errors = []
        for key, value in config.items():
            if isinstance(value, str) and ('path' in key.lower() or 'dir' in key.lower()):
                path = Path(value)
                if not path.exists():
                    try:
                        path.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        errors.append(f"Cannot create path {value}: {e}")
            elif isinstance(value, dict):
                errors.extend(ConfigValidator.verify_paths(value))
        return errors
