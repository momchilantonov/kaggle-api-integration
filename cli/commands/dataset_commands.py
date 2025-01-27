import click
from pathlib import Path
from workflows.dataset_workflows.download_operations import DatasetDownloadManager
from workflows.dataset_workflows.upload_operations import DatasetUploadManager
from src.handlers.data_handlers import DataHandler
from src.utils.error_handlers import handle_api_errors, CLIError

@click.group()
def dataset():
    """Dataset operations"""
    pass

@dataset.command()
@click.argument('dataset_name')
@click.option('--output', '-o', default='data/datasets', help='Output directory')
@click.option('--unzip/--no-unzip', default=True, help='Unzip downloaded files')
@handle_api_errors
def download(dataset_name, output, unzip):
    """Download a dataset"""
    try:
        manager = DatasetDownloadManager()

        with click.progressbar(
            length=100,
            label=f'Downloading dataset {dataset_name}'
        ) as bar:
            path = manager.download_dataset(dataset_name, Path(output), unzip=unzip)
            bar.update(100)

        click.echo(click.style(f"\nSuccessfully downloaded dataset to: {path}", fg='green'))

        # Show dataset statistics if available
        try:
            stats = manager.get_dataset_stats(path)
            click.echo("\nDataset Statistics:")
            for file_name, file_stats in stats.items():
                click.echo(click.style(f"\n{file_name}:", fg='blue'))
                click.echo(f"Rows: {file_stats['rows']:,}")
                click.echo(f"Columns: {file_stats['columns']}")
                click.echo(f"Size: {file_stats['memory_usage']:.2f} MB")
        except Exception as e:
            click.echo(click.style(f"\nNote: Could not calculate dataset statistics: {str(e)}", fg='yellow'))

    except Exception as e:
        raise CLIError(f"Error downloading dataset: {str(e)}")

@dataset.command()
@click.argument('path')
@click.option('--name', required=True, help='Dataset name')
@click.option('--description', help='Dataset description')
@click.option('--public/--private', default=True, help='Dataset visibility')
@handle_api_errors
def upload(path, name, description, public):
    """Upload a dataset"""
    try:
        manager = DatasetUploadManager()
        path = Path(path)

        # Validate path exists
        if not path.exists():
            raise CLIError(f"Path not found: {path}")

        # Count files to be uploaded
        files = list(path.glob('**/*'))
        if not files:
            raise CLIError(f"No files found in {path}")

        # Show upload summary
        click.echo("\nUpload Summary:")
        click.echo(f"Dataset Name: {name}")
        click.echo(f"Description: {description or 'N/A'}")
        click.echo(f"Visibility: {'Public' if public else 'Private'}")
        click.echo(f"Files to upload: {len(files)}")

        # Confirm upload
        if not click.confirm('\nProceed with upload?'):
            click.echo("Upload cancelled.")
            return

        # Create metadata
        metadata = manager.create_metadata(
            title=name,
            description=description or name,
            licenses=[{"name": "CC0-1.0"}],
            keywords=[]
        )

        # Upload with progress bar
        with click.progressbar(
            length=len(files),
            label='Uploading dataset'
        ) as bar:
            def progress_callback(current, total):
                bar.update(current)

            result = manager.upload_dataset(
                path,
                metadata,
                public=public,
                progress_callback=progress_callback
            )

        click.echo(click.style("\nDataset uploaded successfully!", fg='green'))
        click.echo(f"Dataset URL: {result.get('url', 'N/A')}")

    except Exception as e:
        raise CLIError(f"Error uploading dataset: {str(e)}")

@dataset.command()
@click.argument('dataset_name')
@click.option('--output', '-o', default='data/datasets/processed', help='Output directory')
@click.option('--handle-missing/--no-handle-missing', default=True, help='Handle missing values')
@handle_api_errors
def process(dataset_name, output, handle_missing):
    """Process a dataset"""
    try:
        manager = DatasetDownloadManager()
        handler = DataHandler()

        # Download and process
        click.echo("Downloading dataset...")
        data_path = manager.download_dataset(dataset_name)

        # Process each CSV file with progress bar
        csv_files = list(Path(data_path).glob('*.csv'))
        if not csv_files:
            raise CLIError(f"No CSV files found in dataset {dataset_name}")

        click.echo("\nProcessing files:")
        for file_path in csv_files:
            click.echo(f"\nProcessing {file_path.name}...")

            # Read and process file
            df = handler.read_csv(file_path)

            if handle_missing:
                with click.progressbar(
                    length=100,
                    label='Handling missing values'
                ) as bar:
                    df = handler.handle_missing_values(df, {
                        'numeric': 'mean',
                        'categorical': 'mode'
                    })
                    bar.update(100)

            # Save processed file
            output_path = Path(output) / f"processed_{file_path.name}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            handler.write_csv(df, output_path)

            # Show quick statistics
            click.echo(f"Rows: {len(df):,}")
            click.echo(f"Columns: {len(df.columns)}")
            click.echo(click.style(f"Saved to: {output_path}", fg='green'))

        click.echo(click.style("\nDataset processing completed!", fg='green'))

    except Exception as e:
        raise CLIError(f"Error processing dataset: {str(e)}")

if __name__ == '__main__':
    dataset()
