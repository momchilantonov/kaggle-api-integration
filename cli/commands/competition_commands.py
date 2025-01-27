import click
from pathlib import Path
from workflows.competition_workflows.submission_operations import CompetitionWorkflowManager
from workflows.competition_workflows.leaderboard_operations import LeaderboardManager

@click.group()
def competition():
    """Competition operations"""
    pass

@competition.command()
@click.option('--page', default=1, help='Page number')
@click.option('--search', help='Search term')
def list(page, search):
    """List available competitions"""
    manager = CompetitionWorkflowManager()
    competitions = manager.list_competitions(page=page, search=search)
    for comp in competitions:
        click.echo(f"{comp['title']} - {comp['deadline']}")

@competition.command()
@click.argument('competition')
@click.option('--output', '-o', default='data/competitions', help='Output directory')
def download(competition, output):
    """Download competition data"""
    manager = CompetitionWorkflowManager()
    path = manager.download_competition_data(competition, Path(output))
    click.echo(f"Downloaded to: {path}")

@competition.command()
@click.argument('competition')
@click.argument('file_path')
@click.option('--message', '-m', help='Submission message')
@click.option('--wait/--no-wait', default=True, help='Wait for scoring')
def submit(competition, file_path, message, wait):
    """Submit to competition"""
    manager = CompetitionWorkflowManager()
    result = manager.submit_predictions(
        competition,
        Path(file_path),
        message or "CLI submission",
        wait_for_scoring=wait
    )
    click.echo(f"Submission result: {result}")

@competition.command()
@click.argument('competition')
@click.option('--download', '-d', is_flag=True, help='Download leaderboard')
@click.option('--output', '-o', help='Output path for downloaded leaderboard')
def leaderboard(competition, download, output):
    """Get competition leaderboard"""
    manager = LeaderboardManager()
    if download:
        path = manager.download_leaderboard(competition, output)
        click.echo(f"Leaderboard downloaded to: {path}")
    else:
        leaderboard = manager.track_leaderboard(competition)
        for entry in leaderboard:
            click.echo(f"{entry['team']} - {entry['score']}")

@competition.command()
@click.argument('competition')
def status(competition):
    """Get competition status"""
    manager = CompetitionWorkflowManager()
    status = manager.get_competition_status(competition)
    click.echo(f"Deadline: {status['deadline']}")
    click.echo(f"Submissions Today: {status['submissions_today']}")
    click.echo(f"Best Score: {status['best_score']}")

if __name__ == '__main__':
    competition()
