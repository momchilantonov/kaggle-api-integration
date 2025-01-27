import click
from pathlib import Path
from workflows.competition_workflows.submission_operations import CompetitionWorkflowManager
from workflows.competition_workflows.leaderboard_operations import LeaderboardManager
from src.utils.error_handlers import handle_api_errors, CLIError

@click.group()
def competition():
    """Competition operations"""
    pass

@competition.command()
@click.option('--page', default=1, help='Page number')
@click.option('--search', help='Search term')
@handle_api_errors
def list(page, search):
    """List available competitions"""
    try:
        manager = CompetitionWorkflowManager()
        competitions = manager.list_competitions(page=page, search=search)

        if not competitions:
            click.echo("No competitions found matching your criteria.")
            return

        for comp in competitions:
            click.echo(click.style(f"\nCompetition: {comp['title']}", fg='green'))
            click.echo(f"Deadline: {comp['deadline']}")
            click.echo(f"Category: {comp.get('category', 'N/A')}")
            click.echo(f"Prize: {comp.get('reward', 'N/A')}")

    except Exception as e:
        raise CLIError(f"Error listing competitions: {str(e)}")

@competition.command()
@click.argument('competition')
@click.option('--output', '-o', default='data/competitions', help='Output directory')
@handle_api_errors
def download(competition, output):
    """Download competition data"""
    try:
        manager = CompetitionWorkflowManager()
        with click.progressbar(
            length=100,
            label=f'Downloading {competition} data'
        ) as bar:
            path = manager.download_competition_data(competition, Path(output))
            bar.update(100)

        click.echo(click.style(f"\nSuccessfully downloaded to: {path}", fg='green'))

    except Exception as e:
        raise CLIError(f"Error downloading competition data: {str(e)}")

@competition.command()
@click.argument('competition')
@click.argument('file_path')
@click.option('--message', '-m', help='Submission message')
@click.option('--wait/--no-wait', default=True, help='Wait for scoring')
@handle_api_errors
def submit(competition, file_path, message, wait):
    """Submit to competition"""
    try:
        manager = CompetitionWorkflowManager()

        # Validate submission file exists
        if not Path(file_path).exists():
            raise CLIError(f"Submission file not found: {file_path}")

        # Show submission confirmation
        if not click.confirm(f'Submit {file_path} to {competition}?'):
            click.echo("Submission cancelled.")
            return

        result = manager.submit_predictions(
            competition,
            Path(file_path),
            message or "CLI submission",
            wait_for_scoring=wait
        )

        if wait:
            click.echo(click.style("\nSubmission Results:", fg='green'))
            click.echo(f"Score: {result.get('score', 'N/A')}")
            click.echo(f"Status: {result.get('status', 'N/A')}")
        else:
            click.echo(click.style("\nSubmission successful!", fg='green'))
            click.echo("Use 'status' command to check scoring.")

    except Exception as e:
        raise CLIError(f"Error submitting to competition: {str(e)}")

@competition.command()
@click.argument('competition')
@click.option('--download', '-d', is_flag=True, help='Download leaderboard')
@click.option('--output', '-o', help='Output path for downloaded leaderboard')
@handle_api_errors
def leaderboard(competition, download, output):
    """Get competition leaderboard"""
    try:
        manager = LeaderboardManager()

        if download:
            path = manager.download_leaderboard(competition, output)
            click.echo(click.style(f"Leaderboard downloaded to: {path}", fg='green'))
        else:
            leaderboard = manager.track_leaderboard(competition)
            click.echo(click.style("\nLeaderboard:", fg='green'))

            # Create formatted table
            click.echo("{:<4} {:<20} {:<10}".format("Rank", "Team", "Score"))
            click.echo("-" * 40)

            for entry in leaderboard:
                click.echo("{:<4} {:<20} {:<10.4f}".format(
                    entry.get('rank', 'N/A'),
                    entry.get('team', 'Unknown')[:20],
                    float(entry.get('score', 0))
                ))

    except Exception as e:
        raise CLIError(f"Error accessing leaderboard: {str(e)}")

@competition.command()
@click.argument('competition')
@handle_api_errors
def status(competition):
    """Get competition status"""
    try:
        manager = CompetitionWorkflowManager()
        status = manager.get_competition_status(competition)

        click.echo(click.style("\nCompetition Status:", fg='green'))
        click.echo(f"Deadline: {status['deadline']}")
        click.echo(f"Submissions Today: {status['submissions_today']}")
        click.echo(f"Best Score: {status['best_score']}")

        # Add warning if deadline is approaching
        if status.get('deadline_warning'):
            click.echo(click.style(
                f"\nWarning: Competition deadline in {status['deadline_warning']} days!",
                fg='yellow'
            ))

    except Exception as e:
        raise CLIError(f"Error getting competition status: {str(e)}")

if __name__ == '__main__':
    competition()
