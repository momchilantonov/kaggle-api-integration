import click
from pathlib import Path
from workflows.kernel_workflows.execution_operations import KernelClient
from workflows.kernel_workflows.management_operations import KernelManagementManager
from src.utils.error_handlers import handle_api_errors, CLIError

@click.group()
def kernel():
    """Kernel operations"""
    pass

@kernel.command()
@click.option('--owner', help='Filter by owner')
@click.option('--search', help='Search term')
@click.option('--language', type=click.Choice(['python', 'r']), help='Filter by language')
@handle_api_errors
def list(owner, search, language):
    """List kernels"""
    try:
        client = KernelClient()
        kernels = client.list_kernels(owner=owner, search=search, language=language)

        if not kernels:
            click.echo("No kernels found matching your criteria.")
            return

        click.echo(click.style("\nAvailable Kernels:", fg='green'))
        click.echo("-" * 60)

        for kernel in kernels:
            click.echo(click.style(f"\nKernel: {kernel['title']}", fg='blue'))
            click.echo(f"Owner: {kernel['owner']}")
            click.echo(f"Language: {kernel['language']}")
            click.echo(f"Last Updated: {kernel.get('lastRunTime', 'N/A')}")
            click.echo(f"Total Votes: {kernel.get('totalVotes', 0)}")

    except Exception as e:
        raise CLIError(f"Error listing kernels: {str(e)}")

@kernel.command()
@click.argument('folder_path')
@click.option('--title', required=True, help='Kernel title')
@click.option('--language', type=click.Choice(['python', 'r']), required=True)
@click.option('--competition', help='Competition source')
@click.option('--dataset', help='Dataset source')
@click.option('--enable-gpu', is_flag=True, help='Enable GPU')
@click.option('--enable-internet', is_flag=True, help='Enable internet')
@handle_api_errors
def push(folder_path, title, language, competition, dataset, enable_gpu, enable_internet):
    """Push a new kernel"""
    try:
        manager = KernelManagementManager()
        folder_path = Path(folder_path)

        # Validate folder exists
        if not folder_path.exists():
            raise CLIError(f"Folder not found: {folder_path}")

        # Show kernel configuration
        click.echo("\nKernel Configuration:")
        click.echo(f"Title: {title}")
        click.echo(f"Language: {language}")
        click.echo(f"GPU Enabled: {enable_gpu}")
        click.echo(f"Internet Enabled: {enable_internet}")
        if competition:
            click.echo(f"Competition: {competition}")
        if dataset:
            click.echo(f"Dataset: {dataset}")

        # Confirm push
        if not click.confirm('\nProceed with kernel push?'):
            click.echo("Push cancelled.")
            return

        metadata = {
            'title': title,
            'language': language,
            'is_private': False,
            'enable_gpu': enable_gpu,
            'enable_internet': enable_internet,
            'competition_sources': [competition] if competition else None,
            'dataset_sources': [dataset] if dataset else None
        }

        with click.progressbar(
            length=100,
            label='Pushing kernel'
        ) as bar:
            result = manager.push_kernel(Path(folder_path), metadata)
            bar.update(100)

        click.echo(click.style("\nKernel pushed successfully!", fg='green'))
        click.echo(f"Kernel URL: {result.get('url', 'N/A')}")

    except Exception as e:
        raise CLIError(f"Error pushing kernel: {str(e)}")

@kernel.command()
@click.argument('owner')
@click.argument('kernel_name')
@click.option('--version', help='Kernel version')
@click.option('--output', '-o', default='data/kernels/scripts', help='Output directory')
@handle_api_errors
def pull(owner, kernel_name, version, output):
    """Pull a kernel"""
    try:
        client = KernelClient()

        with click.progressbar(
            length=100,
            label=f'Pulling kernel {owner}/{kernel_name}'
        ) as bar:
            path = client.pull_kernel(
                owner,
                kernel_name,
                version=version,
                path=Path(output)
            )
            bar.update(100)

        click.echo(click.style(f"\nKernel pulled successfully to: {path}", fg='green'))

        # Show pulled files
        files = list(path.glob('**/*'))
        click.echo("\nPulled files:")
        for file in files:
            click.echo(f"- {file.relative_to(path)}")

    except Exception as e:
        raise CLIError(f"Error pulling kernel: {str(e)}")

@kernel.command()
@click.argument('owner')
@click.argument('kernel_name')
@handle_api_errors
def status(owner, kernel_name):
    """Get kernel status"""
    try:
        client = KernelClient()
        status = client.get_kernel_status(owner, kernel_name)

        click.echo(click.style("\nKernel Status:", fg='green'))
        click.echo(f"Status: {status['status']}")

        if status['status'] == 'complete':
            click.echo(click.style("✓ Execution completed successfully", fg='green'))
        elif status['status'] == 'error':
            click.echo(click.style(f"✗ Error: {status.get('error', 'Unknown error')}", fg='red'))
        elif status['status'] == 'running':
            click.echo(click.style("⟳ Kernel is currently running", fg='yellow'))

        if 'runtime' in status:
            click.echo(f"Runtime: {status['runtime']} seconds")

    except Exception as e:
        raise CLIError(f"Error getting kernel status: {str(e)}")

@kernel.command()
@click.argument('owner')
@click.argument('kernel_name')
@handle_api_errors
def versions(owner, kernel_name):
    """List kernel versions"""
    try:
        manager = KernelManagementManager()
        versions = manager.list_kernel_versions(owner, kernel_name)

        if not versions:
            click.echo("No versions found for this kernel.")
            return

        click.echo(click.style("\nKernel Versions:", fg='green'))
        click.echo("-" * 60)

        for version in versions:
            click.echo(click.style(f"\nVersion: {version['version']}", fg='blue'))
            click.echo(f"Created: {version['created']}")
            click.echo(f"Status: {version['status']}")
            if version.get('message'):
                click.echo(f"Message: {version['message']}")

    except Exception as e:
        raise CLIError(f"Error listing kernel versions: {str(e)}")

@kernel.command()
@click.argument('owner')
@click.argument('kernel_name')
@click.option('--include-data', is_flag=True, help='Include data files')
@handle_api_errors
def backup(owner, kernel_name, include_data):
    """Create kernel backup"""
    try:
        manager = KernelManagementManager()

        click.echo(f"Creating backup for kernel: {owner}/{kernel_name}")
        if include_data:
            click.echo("Including associated data files...")

        with click.progressbar(
            length=100,
            label='Creating backup'
        ) as bar:
            backup_path = manager.create_kernel_backup(kernel_name, include_data)
            bar.update(100)

        click.echo(click.style(f"\nBackup created successfully at: {backup_path}", fg='green'))

        # Show backup contents
        backup_files = list(backup_path.glob('**/*'))
        click.echo("\nBackup contents:")
        for file in backup_files:
            click.echo(f"- {file.relative_to(backup_path)}")

    except Exception as e:
        raise CLIError(f"Error creating kernel backup: {str(e)}")

if __name__ == '__main__':
    kernel()
