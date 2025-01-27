import click
from pathlib import Path
from workflows.dataset_workflows.download_operations import DatasetDownloadManager
from workflows.dataset_workflows.upload_operations import DatasetUploadManager
from src.handlers.data_handlers import DataHandler

@click.group()
def dataset():
    """Dataset operations"""
    pass

@dataset.command()
@click.argument('dataset_name')
@click.option('--output', '-o', default='data/datasets', help='Output directory')
@click.option('--unzip/--no-unzip', default=True, help='Unzip downloaded files')
def download(dataset_name, output, unzip):
    """Download a dataset"""
    try:
        manager = DatasetDownloadManager()
        path = manager.download_dataset(dataset_name, Path(output), unzip=unzip)
        click.echo(f"Successfully downloaded dataset to: {path}")
    except Exception as e:
        click.echo(f"Error downloading dataset: {str(e)}", err=True)

@dataset.command()
@click.argument('path')
@click.option('--name', required=True, help='Dataset name')
@click.option('--description', help='Dataset description')
@click.option('--public/--private', default=True, help='Dataset visibility')
def upload(path, name, description, public):
    """Upload a dataset"""
    try:
        manager = DatasetUploadManager()

        # Create metadata
        metadata = manager.create_metadata(
            title=name,
            description=description or name,
            licenses=[{"name": "CC0-1.0"}],
            keywords=[]
        )

        # Upload dataset
        result = manager.upload_dataset(Path(path), metadata, public=public)
        click.echo(f"Successfully uploaded dataset: {result}")
    except Exception as e:
        click.echo(f"Error uploading dataset: {str(e)}", err=True)

@dataset.command()
@click.argument('dataset_name')
@click.option('--output', '-o', default='data/datasets/processed', help='Output directory')
@click.option('--handle-missing/--no-handle-missing', default=True, help='Handle missing values')
def process(dataset_name, output, handle_missing):
    """Process a dataset"""
    try:
        manager = DatasetDownloadManager()
        handler = DataHandler()

        # Download and process
        data_path = manager.download_dataset(dataset_name)

        # Process each CSV file
        for file_path in Path(data_path).glob('*.csv'):
            df = handler.read_csv(file_path)

            if handle_missing:
                df = handler.handle_missing_values(df, {
                    'numeric': 'mean',
                    'categorical': 'mode'
                })

            # Save processed file
            output_path = Path(output) / f"processed_{file_path.name}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            handler.write_csv(df, output_path)

        click.echo(f"Successfully processed dataset to: {output}")
    except Exception as e:
        click.echo(f"Error processing dataset: {str(e)}", err=True)

if __name__ == '__main__':
    dataset()
