import click
import logging
from pathlib import Path
import sys
import json

# Import command groups
from commands.competition_commands import competition
from commands.dataset_commands import dataset
from commands.model_commands import model
from commands.kernel_commands import kernel

# Import configuration validator and settings
from src.utils.config_validator import ConfigValidator
from config.settings import setup_logger, validate_environment, API_ENDPOINTS

# Import rate limiting
from src.utils.request_manager import RequestManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class KaggleAPICLI:
    def __init__(self):
        self.config_validator = ConfigValidator()
        self.request_manager = None
        self.rate_limit_config = {
            'calls': 10,  # Default rate limit
            'period': 60  # Default period in seconds
        }

    def validate_configs(self):
        """Validate all configuration files at startup"""
        try:
            config_dir = Path('operational_configs')
            validation_results = self.config_validator.validate_all_configs(config_dir)

            # Check for and create required paths
            for category, configs in validation_results.items():
                for config_name, config_data in configs.items():
                    path_errors = self.config_validator.verify_paths(config_data)
                    if path_errors:
                        logger.warning(f"Path issues in {category}/{config_name}: {path_errors}")

            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            return False

    def init_rate_limiter(self, credentials):
        """Initialize rate limiter with credentials"""
        try:
            self.request_manager = RequestManager(
                base_url=API_ENDPOINTS['base_url'],
                auth=(credentials['username'], credentials['key']),
                rate_limit=self.rate_limit_config['calls'],
                rate_period=self.rate_limit_config['period']
            )
            return True
        except Exception as e:
            logger.error(f"Failed to initialize rate limiter: {str(e)}")
            return False

    def load_rate_limit_state(self):
        """Load saved rate limit state if exists"""
        state_file = Path('.kaggle_rate_limit_state')
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                self.rate_limit_config.update(state)
            except Exception as e:
                logger.warning(f"Could not load rate limit state: {e}")

    def save_rate_limit_state(self):
        """Save current rate limit state"""
        state_file = Path('.kaggle_rate_limit_state')
        try:
            with open(state_file, 'w') as f:
                json.dump(self.rate_limit_config, f)
        except Exception as e:
            logger.warning(f"Could not save rate limit state: {e}")

@click.group()
@click.option('--debug/--no-debug', default=False, help='Enable debug mode')
@click.option('--rate-limit', type=int, help='Set custom rate limit')
@click.option('--rate-period', type=int, help='Set custom rate period (seconds)')
@click.pass_context
def cli(ctx, debug, rate_limit, rate_period):
    """Kaggle API Integration CLI"""
    try:
        # Initialize CLI handler
        cli_handler = KaggleAPICLI()
        ctx.obj = cli_handler

        # Load saved rate limit state
        cli_handler.load_rate_limit_state()

        # Update rate limits if provided
        if rate_limit:
            cli_handler.rate_limit_config['calls'] = rate_limit
        if rate_period:
            cli_handler.rate_limit_config['period'] = rate_period

        # Validate environment variables
        credentials = validate_environment()

        # Validate configurations
        if not cli_handler.validate_configs():
            click.echo(click.style(
                "Error: Configuration validation failed. Please check logs for details.",
                fg='red'
            ))
            sys.exit(1)

        # Initialize rate limiter
        if not cli_handler.init_rate_limiter(credentials):
            click.echo(click.style(
                "Error: Failed to initialize rate limiter. Please check logs for details.",
                fg='red'
            ))
            sys.exit(1)

        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
            click.echo(click.style("Debug mode enabled", fg='yellow'))

        # Save current rate limit configuration
        cli_handler.save_rate_limit_state()

        # Show rate limit info
        click.echo(click.style(
            f"\nRate Limit Configuration:",
            fg='blue'
        ))
        click.echo(f"Requests: {cli_handler.rate_limit_config['calls']} per {cli_handler.rate_limit_config['period']} seconds")

    except Exception as e:
        click.echo(click.style(f"Initialization error: {str(e)}", fg='red'))
        sys.exit(1)

# Add command groups
cli.add_command(competition)
cli.add_command(dataset)
cli.add_command(model)
cli.add_command(kernel)

@cli.command()
@click.pass_context
def rate_info(ctx):
    """Show current rate limit information"""
    cli_handler = ctx.obj
    if cli_handler.request_manager:
        click.echo(click.style("\nCurrent Rate Limit Status:", fg='blue'))
        click.echo(f"Requests made: {cli_handler.request_manager.request_count}")
        click.echo(f"Requests remaining: {cli_handler.rate_limit_config['calls'] - cli_handler.request_manager.request_count}")
        click.echo(f"Period: {cli_handler.rate_limit_config['period']} seconds")
    else:
        click.echo(click.style("Rate limiter not initialized", fg='yellow'))

if __name__ == '__main__':
    cli()
