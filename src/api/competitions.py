from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

from src.utils.error_handlers import handle_api_errors, validate_auth
from .kaggle_client import KaggleAPIClient

@dataclass
class SubmissionMetadata:
    message: str
    description: Optional[str] = None
    quiet: bool = False

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}

class CompetitionClient:
    def __init__(self, client: KaggleAPIClient):
        self.client = client

    @handle_api_errors
    @validate_auth
    def list_competitions(
        self,
        search: Optional[str] = None,
        category: Optional[str] = None,
        page: int = 1,
        **kwargs
    ) -> List[Dict]:
        params = {'search': search, 'category': category, 'page': page, **kwargs}
        response = self.client.get('competitions/list', params=params)
        return response.json()

    @handle_api_errors
    @validate_auth
    def get_competition_details(self, competition: str) -> Dict:
        return self.client.get(
            'competitions/details',
            params={'id': competition}
        ).json()

    @handle_api_errors
    @validate_auth
    def download_competition_files(
        self,
        competition: str,
        path: Optional[Union[str, Path]] = None,
        file_name: Optional[str] = None
    ) -> Path:
        path = Path(path) if path else Path.cwd()
        path.mkdir(parents=True, exist_ok=True)

        params = {'id': competition}
        if file_name:
            params['fileName'] = file_name

        response = self.client.get('competitions/download', params=params, stream=True)

        if file_name:
            return self.client.download_file(response.url, path / file_name)
        else:
            return self.client.download_file(response.url, path / f"{competition}.zip")

    @handle_api_errors
    @validate_auth
    def submit_to_competition(
        self,
        competition: str,
        file_path: Union[str, Path],
        metadata: SubmissionMetadata
    ) -> Dict:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Submission file not found: {file_path}")

        return self.client.post(
            'competitions/submit',
            files={'file': open(file_path, 'rb')},
            data={'id': competition, **metadata.to_dict()}
        ).json()

    @handle_api_errors
    @validate_auth
    def get_submission_status(
        self,
        competition: str,
        submission_id: str
    ) -> Dict:
        return self.client.get(
            'competitions/submissions/status',
            params={'id': competition, 'submissionId': submission_id}
        ).json()

    def wait_for_scoring(
        self,
        competition: str,
        submission_id: str,
        timeout: int = 3600,
        check_interval: int = 10
    ) -> Dict:
        import time
        start_time = time.time()

        while True:
            status = self.get_submission_status(competition, submission_id)

            if status.get('status') == 'complete':
                return status

            if status.get('status') == 'failed':
                raise RuntimeError(f"Submission failed: {status.get('errorMessage')}")

            if time.time() - start_time > timeout:
                raise TimeoutError(f"Scoring not completed after {timeout} seconds")

            time.sleep(check_interval)

    @handle_api_errors
    @validate_auth
    def download_leaderboard(
        self,
        competition: str,
        path: Optional[Union[str, Path]] = None
    ) -> Path:
        path = Path(path) if path else Path.cwd()
        path.mkdir(parents=True, exist_ok=True)

        response = self.client.get(
            'competitions/leaderboard/download',
            params={'id': competition}
        )

        leaderboard_path = path / f"{competition}_leaderboard.csv"
        leaderboard_path.write_bytes(response.content)
        return leaderboard_path

    @handle_api_errors
    @validate_auth
    def get_submission_history(
        self,
        competition: str,
        page: int = 1
    ) -> List[Dict]:
        return self.client.get(
            'competitions/submissions/list',
            params={'id': competition, 'page': page}
        ).json()
