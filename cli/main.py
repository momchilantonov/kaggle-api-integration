import click
import logging
from pathlib import Path

# Import command groups
from commands.competition_commands import competition
from commands.dataset_commands import dataset
from commands.model_commands import model
from commands.kernel_commands import kernel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@click.group()
@click.option('--debug/--no-debug', default=False, help='Enable debug mode')
def cli(debug):
    """Kaggle API Integration CLI"""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

# Add command groups
cli.add_command(competition)
cli.add_command(dataset)
cli.add_command(model)
cli.add_command(kernel)

if __name__ == '__main__':
    cli()
