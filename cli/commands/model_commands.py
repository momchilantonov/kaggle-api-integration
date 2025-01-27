import click
from pathlib import Path
from workflows.model_workflows.download_operations import ModelDownloadManager
from workflows.model_workflows.upload_operations import ModelUploadManager
from workflows.integrations.model_training import ModelTrainingPipeline

@click.group()
def model():
    """Model operations"""
    pass

@model.command()
@click.argument('dataset_name')
@click.option('--model-type', '-t', default='random_forest', help='Model type')
@click.option('--target', required=True, help='Target column')
@click.option('--output', '-o', default='data/models/custom', help='Output path')
def train(dataset_name, model_type, target, output):
    """Train a model"""
    pipeline = ModelTrainingPipeline()
    result = pipeline.train_model(
        dataset_name=dataset_name,
        model_type=model_type,
        target_column=target
    )
    click.echo(f"Model saved to: {result['model_path']}")
    click.echo("\nMetrics:")
    for metric, value in result['metrics'].items():
        click.echo(f"{metric}: {value:.4f}")

@model.command()
@click.argument('model_path')
@click.option('--name', required=True, help='Model name')
@click.option('--description', help='Model description')
@click.option('--framework', default='sklearn', help='Framework used')
def upload(model_path, name, description, framework):
    """Upload a model"""
    manager = ModelUploadManager()

    # Prepare model directory
    upload_dir = manager.prepare_model_upload(
        Path(model_path),
        framework=framework,
        include_artifacts=True
    )

    # Create metadata
    metadata = manager.create_metadata(
        name=name,
        version_name="v1.0",
        description=description or name,
        framework=framework,
        task_ids=["classification"]
    )

    # Upload
    result = manager.upload_model(upload_dir, metadata)
    click.echo(f"Upload result: {result}")

@model.command()
@click.argument('owner')
@click.argument('model_name')
@click.option('--version', help='Model version')
@click.option('--output', '-o', default='data/models/downloaded', help='Output directory')
def download(owner, model_name, version, output):
    """Download a model"""
    manager = ModelDownloadManager()
    path = manager.download_model(owner, model_name, version, Path(output))
    click.echo(f"Downloaded to: {path}")

@model.command()
@click.argument('model_path')
@click.argument('data_path')
@click.option('--target', required=True, help='Target column')
def evaluate(model_path, data_path, target):
    """Evaluate a model"""
    pipeline = ModelTrainingPipeline()
    metrics = pipeline.evaluate_model(
        model_path=Path(model_path),
        data_path=Path(data_path),
        target_column=target
    )
    click.echo("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        click.echo(f"{metric}: {value:.4f}")

@model.command()
@click.argument('owner')
@click.argument('model_name')
def versions(owner, model_name):
    """List model versions"""
    manager = ModelDownloadManager()
    versions = manager.list_model_versions(owner, model_name)
    for version in versions:
        click.echo(f"Version: {version['version_number']} - {version['created']}")

if __name__ == '__main__':
    model()
