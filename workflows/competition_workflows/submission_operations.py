from pathlib import Path
import yaml
from typing import Optional, List, Dict, Union
import logging
import time
import pandas as pd

from src.api.kaggle_client import KaggleAPIClient
from src.api.competitions import CompetitionClient, SubmissionMetadata
from src.handlers.data_handlers import DataHandler
from src.utils.helpers import timer, retry_on_exception

logger = logging.getLogger(__name__)

class CompetitionWorkflowManager:
    def __init__(self):
        """Initialize the competition workflow manager"""
        self.kaggle_client = KaggleAPIClient()
        self.competition_client = CompetitionClient(self.kaggle_client)
        self.data_handler = DataHandler()
        self._load_configs()

    def _load_configs(self):
        """Load operational configurations"""
        try:
            with open('operational_configs/competition_configs/submission_rules.yaml', 'r') as f:
                self.submission_config = yaml.safe_load(f)
            with open('operational_configs/competition_configs/competition_params.yaml', 'r') as f:
                self.competition_config = yaml.safe_load(f)
            logger.info("Successfully loaded competition configurations")
        except Exception as e:
            logger.error(f"Error loading configurations: {str(e)}")
            raise

    @timer
    @retry_on_exception(retries=3, delay=1)
    def download_competition_data(
        self,
        competition: str,
        file_name: Optional[str] = None,
        custom_path: Optional[Path] = None
    ) -> Path:
        """
        Download competition data

        Args:
            competition: Competition name
            file_name: Specific file to download
            custom_path: Optional custom download path

        Returns:
            Path to downloaded data
        """
        try:
            # Determine download path
            base_path = custom_path or Path(self.competition_config['data_paths']['default'])
            download_path = base_path / competition
            download_path.mkdir(parents=True, exist_ok=True)

            # Download data
            data_path = self.competition_client.download_competition_files(
                competition,
                path=download_path,
                file_name=file_name
            )

            logger.info(f"Successfully downloaded competition data to {data_path}")
            return data_path

        except Exception as e:
            logger.error(f"Error downloading competition data: {str(e)}")
            raise

    @timer
    def prepare_submission(
        self,
        predictions_df: pd.DataFrame,
        competition: str
    ) -> Path:
        """
        Prepare competition submission

        Args:
            predictions_df: DataFrame with predictions
            competition: Competition name

        Returns:
            Path to prepared submission file
        """
        try:
            # Get submission rules for competition
            rules = self.submission_config['submission_settings'].get(
                competition,
                self.submission_config['submission_settings']['default']
            )

            # Validate submission format
            is_valid, errors = self.data_handler.validate_submission_format(
                predictions_df,
                required_columns=rules['required_columns'],
                column_types=rules['column_types']
            )

            if not is_valid:
                raise ValueError(f"Invalid submission format: {errors}")

            # Save submission file
            submission_dir = Path(self.competition_config['data_paths']['submissions'])
            submission_dir.mkdir(parents=True, exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            submission_path = submission_dir / f"submission_{timestamp}.csv"

            self.data_handler.write_csv(
                predictions_df,
                submission_path,
                index=False
            )

            logger.info(f"Prepared submission file: {submission_path}")
            return submission_path

        except Exception as e:
            logger.error(f"Error preparing submission: {str(e)}")
            raise

    @timer
    def submit_predictions(
        self,
        competition: str,
        submission_path: Path,
        message: str,
        wait_for_scoring: bool = True
    ) -> Dict:
        """
        Submit predictions to competition

        Args:
            competition: Competition name
            submission_path: Path to submission file
            message: Submission description
            wait_for_scoring: Whether to wait for scoring

        Returns:
            Submission results
        """
        try:
            # Create submission metadata
            metadata = SubmissionMetadata(
                message=message,
                description=f"Submission at {time.strftime('%Y-%m-%d %H:%M:%S')}",
                quiet=False
            )

            # Submit
            result = self.competition_client.submit_to_competition(
                competition,
                submission_path,
                metadata
            )

            if wait_for_scoring:
                result = self.wait_for_scoring(
                    competition,
                    result['id'],
                    timeout=self.submission_config.get('scoring_timeout', 3600)
                )

            logger.info(f"Successfully submitted to competition: {result}")
            return result

        except Exception as e:
            logger.error(f"Error submitting to competition: {str(e)}")
            raise

    def wait_for_scoring(
        self,
        competition: str,
        submission_id: str,
        timeout: int = 3600,
        check_interval: int = 10
    ) -> Dict:
        """
        Wait for submission scoring

        Args:
            competition: Competition name
            submission_id: Submission ID
            timeout: Maximum time to wait in seconds
            check_interval: Time between status checks

        Returns:
            Final submission status
        """
        start_time = time.time()
        while True:
            try:
                status = self.competition_client.get_submission_status(
                    competition,
                    submission_id
                )

                if status.get('status') == 'complete':
                    logger.info(f"Submission {submission_id} scored successfully")
                    return status

                if status.get('status') == 'failed':
                    error_msg = status.get('errorMessage', 'Unknown error')
                    raise RuntimeError(f"Submission failed: {error_msg}")

                if time.time() - start_time > timeout:
                    raise TimeoutError(
                        f"Scoring not completed after {timeout} seconds"
                    )

                time.sleep(check_interval)

            except Exception as e:
                logger.error(f"Error checking submission status: {str(e)}")
                raise

    @timer
    def download_leaderboard(
        self,
        competition: str,
        custom_path: Optional[Path] = None
    ) -> Path:
        """
        Download competition leaderboard

        Args:
            competition: Competition name
            custom_path: Optional custom download path

        Returns:
            Path to leaderboard file
        """
        try:
            leaderboard_path = self.competition_client.download_leaderboard(
                competition,
                path=custom_path or Path(self.competition_config['data_paths']['leaderboards'])
            )

            logger.info(f"Downloaded leaderboard to {leaderboard_path}")
            return leaderboard_path

        except Exception as e:
            logger.error(f"Error downloading leaderboard: {str(e)}")
            raise

def main():
    """Example usage of competition workflows"""
    try:
        # Initialize manager
        manager = CompetitionWorkflowManager()

        # Download competition data
        data_path = manager.download_competition_data("titanic")
        print(f"\nDownloaded competition data to: {data_path}")

        # Example of preparing and submitting predictions
        predictions = pd.DataFrame({
            'PassengerId': range(892, 1310),
            'Survived': [1, 0] * 209  # Dummy predictions
        })

        # Prepare submission
        submission_path = manager.prepare_submission(
            predictions,
            "titanic"
        )
        print(f"\nPrepared submission file: {submission_path}")

        # Submit predictions
        result = manager.submit_predictions(
            "titanic",
            submission_path,
            "Test submission",
            wait_for_scoring=True
        )
        print(f"\nSubmission result: {result}")

        # Download leaderboard
        leaderboard_path = manager.download_leaderboard("titanic")
        print(f"\nDownloaded leaderboard to: {leaderboard_path}")

    except Exception as e:
        print(f"Error in competition workflow: {str(e)}")

if __name__ == "__main__":
    main()
