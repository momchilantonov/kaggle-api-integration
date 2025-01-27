import click
from pathlib import Path
from workflows.kernel_workflows.execution_operations import KernelClient
from workflows.kernel_workflows.management_operations import KernelManagementManager

@click.group()
def kernel():
    """Kernel operations"""
    pass

@kernel.command()
@click.option('--owner', help='Filter by owner')
@click.option('--search', help='Search term')
@click.option('--language', type=click.Choice(['python', 'r']), help='Filter by language')
def list(owner, search, language):
    """List kernels"""
    client = KernelClient()
    kernels = client.list_kernels(owner=owner, search=search, language=language)
    for kernel in kernels:
        click.echo(f"{kernel['owner']}/{kernel['title']} - {kernel['language']}")

@kernel.command()
@click.argument('folder_path')
@click.option('--title', required=True, help='Kernel title')
@click.option('--language', type=click.Choice(['python', 'r']), required=True)
@click.option('--competition', help='Competition source')
@click.option('--dataset', help='Dataset source')
@click.option('--enable-gpu', is_flag=True, help='Enable GPU')
@click.option('--enable-internet', is_flag=True, help='Enable internet')
def push(folder_path, title, language, competition, dataset, enable_gpu, enable_internet):
    """Push a new kernel"""
    manager = KernelManagementManager()
    metadata = {
        'title': title,
        'language': language,
        'is_private': False,
        'enable_gpu': enable_gpu,
        'enable_internet': enable_internet,
        'competition_sources': [competition] if competition else None,
        'dataset_sources': [dataset] if dataset else None
    }
    result = manager.push_kernel(Path(folder_path), metadata)
    click.echo(f"Push result: {result}")

@kernel.command()
@click.argument('owner')
@click.argument('kernel_name')
@click.option('--version', help='Kernel version')
@click.option('--output', '-o', default='data/kernels/scripts', help='Output directory')
def pull(owner, kernel_name, version, output):
    """Pull a kernel"""
    client = KernelClient()
    path = client.pull_kernel(owner, kernel_name, version, Path(output))
    click.echo(f"Pulled to: {path}")

@kernel.command()
@click.argument('owner')
@click.argument('kernel_name')
def status(owner, kernel_name):
    """Get kernel status"""
    client = KernelClient()
    status = client.get_kernel_status(owner, kernel_name)
    click.echo(f"Status: {status['status']}")
    if 'error' in status:
        click.echo(f"Error: {status['error']}")

@kernel.command()
@click.argument('owner')
@click.argument('kernel_name')
def versions(owner, kernel_name):
    """List kernel versions"""
    manager = KernelManagementManager()
    versions = manager.list_kernel_versions(owner, kernel_name)
    for version in versions:
        click.echo(f"Version {version['version']}: {version['created']}")

@kernel.command()
@click.argument('owner')
@click.argument('kernel_name')
@click.option('--include-data', is_flag=True, help='Include data files')
def backup(owner, kernel_name, include_data):
    """Create kernel backup"""
    manager = KernelManagementManager()
    backup_path = manager.create_kernel_backup(kernel_name, include_data)
    click.echo(f"Backup created at: {backup_path}")

if __name__ == '__main__':
    kernel()
