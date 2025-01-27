import click
from pathlib import Path
from workflows.model_workflows.download_operations import ModelDownloadManager
from workflows.model_workflows.upload_operations import ModelUploadManager
from workflows.integrations.model_training import ModelTrainingPipeline
from src.utils.error_handlers import handle_api_errors, CLIError

@click.group()
def model():
    """Model operations"""
    pass

@model.command()
@click.argument('dataset_name')
@click.option('--model-type', '-t', default='random_forest', help='Model type')
@click.option('--target', required=True, help='Target column')
@click.option('--output', '-o', default='data/models/custom', help='Output path')
@handle_api_errors
def train(dataset_name, model_type, target, output):
    """Train a model"""
    try:
        pipeline = ModelTrainingPipeline()

        click.echo(click.style("\nStarting model training:", fg='green'))
        click.echo(f"Dataset: {dataset_name}")
        click.echo(f"Model Type: {model_type}")
        click.echo(f"Target Column: {target}")

        with click.progressbar(
            length=100,
            label='Training model'
        ) as bar:
            result = pipeline.train_model(
                dataset_name=dataset_name,
                model_type=model_type,
                target_column=target
            )
            bar.update(100)

        # Display results
        click.echo(click.style("\nTraining completed successfully!", fg='green'))
        click.echo(f"\nModel saved to: {result['model_path']}")

        click.echo(click.style("\nModel Metrics:", fg='blue'))
        for metric, value in result['metrics'].items():
            click.echo(f"{metric.capitalize()}: {value:.4f}")

        if result.get('upload_result'):
            click.echo(click.style("\nModel Upload Info:", fg='blue'))
            click.echo(f"Model URL: {result['upload_result'].get('url', 'N/A')}")

    except Exception as e:
        raise CLIError(f"Error training model: {str(e)}")

@model.command()
@click.argument('model_path')
@click.option('--name', required=True, help='Model name')
@click.option('--description', help='Model description')
@click.option('--framework', default='sklearn', help='Framework used')
@handle_api_errors
def upload(model_path, name, description, framework):
    """Upload a model"""
    try:
        manager = ModelUploadManager()

        # Validate model path
        model_path = Path(model_path)
        if not model_path.exists():
            raise CLIError(f"Model path not found: {model_path}")

        # Prepare model directory with progress feedback
        click.echo("\nPreparing model for upload...")
        click.echo(f"Framework: {framework}")

        with click.progressbar(
            length=100,
            label='Preparing model'
        ) as bar:
            upload_dir = manager.prepare_model_upload(
                model_path,
                framework=framework,
                include_artifacts=True
            )
            bar.update(100)

        # Create metadata with model card
        click.echo("\nCreating model metadata...")
        metadata = manager.create_metadata(
            name=name,
            version_name="v1.0",
            description=description or name,
            framework=framework,
            task_ids=["classification"]  # Could be made configurable
        )

        # Confirm upload
        click.echo("\nUpload Summary:")
        click.echo(f"Name: {name}")
        click.echo(f"Description: {description or 'N/A'}")
        click.echo(f"Framework: {framework}")

        if not click.confirm('\nProceed with upload?'):
            click.echo("Upload cancelled.")
            return

        # Upload model
        with click.progressbar(
            length=100,
            label='Uploading model'
        ) as bar:
            result = manager.upload_model(upload_dir, metadata)
            bar.update(100)

        click.echo(click.style("\nModel uploaded successfully!", fg='green'))
        click.echo(f"Model URL: {result.get('url', 'N/A')}")

    except Exception as e:
        raise CLIError(f"Error uploading model: {str(e)}")

@model.command()
@click.argument('owner')
@click.argument('model_name')
@click.option('--version', help='Model version')
@click.option('--output', '-o', default='data/models/downloaded', help='Output directory')
@handle_api_errors
def download(owner, model_name, version, output):
    """Download a model"""
    try:
        manager = ModelDownloadManager()

        with click.progressbar(
            length=100,
            label=f'Downloading model {owner}/{model_name}'
        ) as bar:
            path = manager.download_model(
                owner,
                model_name,
                version=version,
                custom_path=Path(output)
            )
            bar.update(100)

        click.echo(click.style(f"\nModel downloaded successfully to: {path}", fg='green'))

        # Show downloaded files
        files = list(path.glob('**/*'))
        click.echo("\nDownloaded files:")
        for file in files:
            click.echo(f"- {file.relative_to(path)}")

        # Try to show model info if available
        try:
            info = manager.get_model_info(path)
            click.echo(click.style("\nModel Information:", fg='blue'))
            for key, value in info.items():
                click.echo(f"{key}: {value}")
        except Exception:
            pass

    except Exception as e:
        raise CLIError(f"Error downloading model: {str(e)}")

@model.command()
@click.argument('model_path')
@click.argument('data_path')
@click.option('--target', required=True, help='Target column')
@handle_api_errors
def evaluate(model_path, data_path, target):
    """Evaluate a model"""
    try:
        pipeline = ModelTrainingPipeline()

        click.echo("\nEvaluating model...")
        with click.progressbar(
            length=100,
            label='Running evaluation'
        ) as bar:
            metrics = pipeline.evaluate_model(
                model_path=Path(model_path),
                data_path=Path(data_path),
                target_column=target
            )
            bar.update(100)

        click.echo(click.style("\nEvaluation Metrics:", fg='green'))
        for metric, value in metrics.items():
            click.echo(f"{metric.capitalize()}: {value:.4f}")

    except Exception as e:
        raise CLIError(f"Error evaluating model: {str(e)}")

@model.command()
@click.argument('owner')
@click.argument('model_name')
@handle_api_errors
def versions(owner, model_name):
    """List model versions"""
    try:
        manager = ModelDownloadManager()
        versions = manager.list_model_versions(owner, model_name)

        if not versions:
            click.echo("No versions found for this model.")
            return

        click.echo(click.style("\nModel Versions:", fg='green'))
        click.echo("-" * 60)

        for version in versions:
            click.echo(click.style(f"\nVersion: {version['version_number']}", fg='blue'))
            click.echo(f"Created: {version['created']}")
            click.echo(f"Framework: {version.get('framework', 'N/A')}")
            click.echo(f"Size: {version.get('size', 'N/A')}")
            if version.get('description'):
                click.echo(f"Description: {version['description']}")

    except Exception as e:
        raise CLIError(f"Error listing model versions: {str(e)}")

if __name__ == '__main__':
    model()
