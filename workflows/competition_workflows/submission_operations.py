from pathlib import Path
from typing import Dict, Optional, Union
import logging
from datetime import datetime

from src.api.competitions import CompetitionClient, SubmissionMetadata
from src.handlers.data_handlers import DataHandler
from src.utils.path_manager import PathManager
from src.utils.error_handlers import handle_api_errors

logger = logging.getLogger(__name__)

class CompetitionWorkflowManager:
    """Manages competition submission workflows"""

    def __init__(self):
        self.competition_client = CompetitionClient()
        self.data_handler = DataHandler()
        self.path_manager = PathManager()
        # Ensure required directories exist
        self.path_manager.ensure_directories()

    @handle_api_errors
    def download_competition_data(
        self,
        competition: str,
        custom_path: Optional[Path] = None
    ) -> Path:
        """Download competition data files"""
        try:
            # Get competition directory path
            comp_path = custom_path or self.path_manager.get_path('competitions', 'data') / competition
            comp_path.mkdir(parents=True, exist_ok=True)

            # Download data
            data_path = self.competition_client.download_competition_files(
                competition=competition,
                path=comp_path
            )

            logger.info(f"Downloaded competition data to {data_path}")
            return data_path

        except Exception as e:
            logger.error(f"Error downloading competition data: {str(e)}")
            raise

    @handle_api_errors
    def submit_predictions(
        self,
        competition: str,
        file_path: Union[str, Path],
        message: str,
        wait_for_scoring: bool = True
    ) -> Dict:
        """Submit predictions to competition"""
        try:
            file_path = Path(file_path)

            # Validate file exists
            if not file_path.exists():
                raise FileNotFoundError(f"Submission file not found: {file_path}")

            # Create backup before submission
            backup_path = self.path_manager.backup_file(
                file_path,
                backup_dir=self.path_manager.get_path('competitions', 'submissions') / 'backups'
            )

            # Create submission metadata
            metadata = SubmissionMetadata(
                message=message,
                description=f"Submission at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                quiet=False
            )

            # Submit
            result = self.competition_client.submit_to_competition(
                competition=competition,
                file_path=file_path,
                metadata=metadata
            )

            if wait_for_scoring:
                result = self.competition_client.wait_for_submission_scoring(
                    competition=competition,
                    submission_id=result['id']
                )

            # Log submission details
            submission_log_path = (
                self.path_manager.get_path('competitions', 'submissions') /
                f"{competition}_submissions.log"
            )
            with open(submission_log_path, 'a') as f:
                f.write(
                    f"{datetime.now()},{file_path.name},{result.get('id')},{result.get('status')}\n"
                )

            return result

        except Exception as e:
            logger.error(f"Error submitting predictions: {str(e)}")
            raise

    @handle_api_errors
    def get_competition_status(self, competition: str) -> Dict:
        """Get competition status and submission history"""
        try:
            # Get competition details
            details = self.competition_client.get_competition_details(competition)

            # Get submission history
            submissions = self.competition_client.get_submission_history(competition)

            # Calculate submissions today
            today = datetime.now().date()
            submissions_today = sum(
                1 for sub in submissions
                if datetime.fromisoformat(sub['date']).date() == today
            )

            # Get best score
            best_score = max(
                (sub['score'] for sub in submissions if sub.get('score')),
                default=None
            )

            status = {
                'deadline': details.get('deadline'),
                'submissions_today': submissions_today,
                'best_score': best_score,
                'total_submissions': len(submissions)
            }

            # Add deadline warning if approaching
            if details.get('deadline'):
                deadline = datetime.fromisoformat(details['deadline'])
                days_remaining = (deadline - datetime.now()).days
                if days_remaining <= 7:
                    status['deadline_warning'] = days_remaining

            return status

        except Exception as e:
            logger.error(f"Error getting competition status: {str(e)}")
            raise

if __name__ == '__main__':
    # Example usage
    manager = CompetitionWorkflowManager()

    # Test competition workflow
    try:
        # Download competition data
        data_path = manager.download_competition_data("titanic")
        print(f"Downloaded data to: {data_path}")

        # Submit predictions (if you have a submission file)
        submission_file = data_path / "submission.csv"
        if submission_file.exists():
            result = manager.submit_predictions(
                "titanic",
                submission_file,
                "Test submission"
            )
            print(f"Submission result: {result}")

        # Get competition status
        status = manager.get_competition_status("titanic")
        print(f"Competition status: {status}")

    except Exception as e:
        print(f"Error: {str(e)}")
