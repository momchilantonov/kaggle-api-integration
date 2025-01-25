from typing import Dict, List, Optional, Union
from pathlib import Path
import time
from dataclasses import dataclass

from .kaggle_client import KaggleAPIClient
from config.settings import setup_logger

logger = setup_logger('kaggle_competitions', 'kaggle_competitions.log')

@dataclass
class SubmissionMetadata:
    """Metadata for competition submissions"""
    message: str
    description: Optional[str] = None
    quiet: bool = False

    def to_dict(self) -> Dict:
        """Convert metadata to dictionary format"""
        return {k: v for k, v in self.__dict__.items() if v is not None}

class CompetitionClient:
    """Client for handling Kaggle competition operations"""

    def __init__(self, client: KaggleAPIClient):
        """Initialize with a KaggleAPIClient instance"""
        self.client = client

    def list_competitions(
        self,
        search: Optional[str] = None,
        category: Optional[str] = None,
        sort_by: Optional[str] = None,
        group: Optional[str] = None,
        page: int = 1,
        page_size: int = 20
    ) -> List[Dict]:
        """
        List competitions on Kaggle with filtering options

        Args:
            search: Search terms
            category: Filter by category
            sort_by: Sort results (prize, deadline, etc)
            group: Filter by group (active, general, entered, etc)
            page: Page number for pagination
            page_size: Number of results per page

        Returns:
            List of competitions matching criteria
        """
        params = {
            'search': search,
            'category': category,
            'sortBy': sort_by,
            'group': group,
            'page': page,
            'pageSize': page_size
        }

        params = {k: v for k, v in params.items() if v is not None}

        response = self.client.get('competitions_list', params=params)
        competitions = response.json()
        logger.info(f"Found {len(competitions)} competitions matching criteria")
        return competitions

    def get_competition_details(self, competition: str) -> Dict:
        """
        Get detailed information about a competition

        Args:
            competition: Competition name/ID

        Returns:
            Competition details
        """
        response = self.client.get(
            'competition_details',
            params={'id': competition}
        )
        return response.json()

    def download_competition_files(
        self,
        competition: str,
        path: Optional[Union[str, Path]] = None,
        file_name: Optional[str] = None,
        quiet: bool = False
    ) -> Path:
        """
        Download competition files

        Args:
            competition: Competition name/ID
            path: Path to save files
            file_name: Specific file to download
            quiet: Whether to suppress progress output

        Returns:
            Path to downloaded file(s)
        """
        path = Path(path) if path else Path.cwd()
        path.mkdir(parents=True, exist_ok=True)

        params = {'id': competition}
        if file_name:
            params['fileName'] = file_name

        response = self.client.get(
            'competition_download',
            params=params,
            stream=True
        )

        if file_name:
            return self.client.download_file(response, path, file_name)
        else:
            zip_path = self.client.download_file(
                response,
                path,
                f"{competition}.zip"
            )

            if not quiet:
                logger.info(f"Downloaded competition files to {zip_path}")
            return zip_path

    def submit_to_competition(
        self,
        competition: str,
        file_path: Union[str, Path],
        metadata: SubmissionMetadata
    ) -> Dict:
        """
        Submit to a competition

        Args:
            competition: Competition name/ID
            file_path: Path to submission file
            metadata: Submission metadata

        Returns:
            Submission results
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Submission file not found: {file_path}")

        # Upload submission
        response = self.client.post(
            'competition_submit',
            files={'file': open(file_path, 'rb')},
            data={
                'id': competition,
                **metadata.to_dict()
            }
        )

        result = response.json()
        if not metadata.quiet:
            logger.info(
                f"Submitted {file_path.name} to competition {competition}"
            )
        return result

    def get_submission_status(
        self,
        competition: str,
        submission_id: str
    ) -> Dict:
        """
        Get status of a competition submission

        Args:
            competition: Competition name/ID
            submission_id: ID of the submission

        Returns:
            Submission status
        """
        response = self.client.get(
            'competition_submission_status',
            params={
                'id': competition,
                'submissionId': submission_id
            }
        )
        return response.json()

    def get_competition_submissions(
        self,
        competition: str,
        page: int = 1,
        page_size: int = 20
    ) -> List[Dict]:
        """
        Get list of submissions for a competition

        Args:
            competition: Competition name/ID
            page: Page number for pagination
            page_size: Number of results per page

        Returns:
            List of submissions
        """
        response = self.client.get(
            'competition_submissions',
            params={
                'id': competition,
                'page': page,
                'pageSize': page_size
            }
        )
        return response.json()

    def wait_for_submission_completion(
        self,
        competition: str,
        submission_id: str,
        timeout: int = 3600,  # 1 hour
        check_interval: int = 10
    ) -> Dict:
        """
        Wait for a submission to complete

        Args:
            competition: Competition name/ID
            submission_id: ID of the submission
            timeout: Maximum time to wait in seconds
            check_interval: Time between checks in seconds

        Returns:
            Final submission status

        Raises:
            TimeoutError: If submission doesn't complete within timeout
            RuntimeError: If submission fails
        """
        start_time = time.time()

        while True:
            status = self.get_submission_status(competition, submission_id)

            if status.get('status') == 'complete':
                logger.info(f"Submission {submission_id} completed")
                return status

            if status.get('status') == 'failed':
                error_msg = status.get('errorMessage', 'Unknown error')
                logger.error(f"Submission failed: {error_msg}")
                raise RuntimeError(f"Submission failed: {error_msg}")

            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Submission not completed after {timeout} seconds"
                )

            time.sleep(check_interval)

    def download_leaderboard(
        self,
        competition: str,
        path: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Download competition leaderboard

        Args:
            competition: Competition name/ID
            path: Path to save the leaderboard

        Returns:
            Path to downloaded leaderboard CSV
        """
        path = Path(path) if path else Path.cwd()
        path.mkdir(parents=True, exist_ok=True)

        response = self.client.get(
            'competition_leaderboard_download',
            params={'id': competition}
        )

        leaderboard_path = path / f"{competition}_leaderboard.csv"
        leaderboard_path.write_bytes(response.content)

        logger.info(f"Downloaded leaderboard to {leaderboard_path}")
        return leaderboard_path
