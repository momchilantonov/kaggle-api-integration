import click
import logging
from pathlib import Path
import sys

# Import command groups
from commands.competition_commands import competition
from commands.dataset_commands import dataset
from commands.model_commands import model
from commands.kernel_commands import kernel

# Import configuration validator
from src.utils.config_validator import ConfigValidator
from config.settings import setup_logger, validate_environment

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def validate_configs():
    """Validate all configuration files at startup"""
    try:
        config_dir = Path('operational_configs')
        validator = ConfigValidator()
        validation_results = validator.validate_all_configs(config_dir)

        # Check for and create required paths
        for category, configs in validation_results.items():
            for config_name, config_data in configs.items():
                path_errors = validator.verify_paths(config_data)
                if path_errors:
                    logger.warning(f"Path issues in {category}/{config_name}: {path_errors}")

        return True
    except Exception as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        return False

@click.group()
@click.option('--debug/--no-debug', default=False, help='Enable debug mode')
def cli(debug):
    """Kaggle API Integration CLI"""
    try:
        # Validate environment variables
        validate_environment()

        # Validate configurations
        if not validate_configs():
            click.echo("Error: Configuration validation failed. Please check logs for details.")
            sys.exit(1)

        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
            click.echo("Debug mode enabled")

    except Exception as e:
        click.echo(f"Initialization error: {str(e)}")
        sys.exit(1)

# Add command groups
cli.add_command(competition)
cli.add_command(dataset)
cli.add_command(model)
cli.add_command(kernel)

if __name__ == '__main__':
    cli()
