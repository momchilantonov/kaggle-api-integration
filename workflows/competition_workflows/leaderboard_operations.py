from pathlib import Path
import yaml
from typing import Optional, List, Dict, Union
import logging
import pandas as pd
from datetime import datetime

from src.api.kaggle_client import KaggleAPIClient
from src.api.competitions import CompetitionClient
from src.handlers.data_handlers import DataHandler
from src.utils.helpers import timer, retry_on_exception

logger = logging.getLogger(__name__)

class LeaderboardManager:
    def __init__(self):
        """Initialize the leaderboard manager"""
        self.kaggle_client = KaggleAPIClient()
        self.competition_client = CompetitionClient(self.kaggle_client)
        self.data_handler = DataHandler()
        self._load_configs()

    def _load_configs(self):
        """Load operational configurations"""
        try:
            with open('operational_configs/competition_configs/competition_params.yaml', 'r') as f:
                self.competition_config = yaml.safe_load(f)
            logger.info("Successfully loaded competition configurations")
        except Exception as e:
            logger.error(f"Error loading configurations: {str(e)}")
            raise

    @timer
    @retry_on_exception(retries=3, delay=1)
    def track_leaderboard(
        self,
        competition: str,
        store_history: bool = True
    ) -> Dict:
        """
        Download and analyze current leaderboard

        Args:
            competition: Competition name
            store_history: Whether to store historical data

        Returns:
            Dictionary with leaderboard analysis
        """
        try:
            # Download current leaderboard
            leaderboard_path = self.competition_client.download_leaderboard(
                competition
            )

            # Read and analyze leaderboard
            leaderboard_df = pd.read_csv(leaderboard_path)
            analysis = self._analyze_leaderboard(leaderboard_df)

            # Store historical data if requested
            if store_history:
                self._store_leaderboard_history(
                    competition,
                    leaderboard_df
                )

            return analysis

        except Exception as e:
            logger.error(f"Error tracking leaderboard: {str(e)}")
            raise

    def _analyze_leaderboard(self, leaderboard_df: pd.DataFrame) -> Dict:
        """Analyze leaderboard data"""
        try:
            analysis = {
                'total_entries': len(leaderboard_df),
                'top_score': leaderboard_df.iloc[0]['Score'],
                'median_score': leaderboard_df['Score'].median(),
                'score_std': leaderboard_df['Score'].std(),
                'timestamp': datetime.now().isoformat()
            }
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing leaderboard: {str(e)}")
            raise

    def _store_leaderboard_history(
        self,
        competition: str,
        leaderboard_df: pd.DataFrame
    ) -> Path:
        """Store leaderboard data for historical tracking"""
        try:
            history_dir = Path(self.competition_config['data_paths']['leaderboard_history'])
            history_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            history_path = history_dir / f"{competition}_leaderboard_{timestamp}.csv"

            leaderboard_df.to_csv(history_path, index=False)
            logger.info(f"Stored leaderboard history: {history_path}")

            return history_path

        except Exception as e:
            logger.error(f"Error storing leaderboard history: {str(e)}")
            raise

    @timer
    def analyze_submission_history(
        self,
        competition: str,
        top_n: int = 10
    ) -> Dict:
        """
        Analyze submission history for a competition

        Args:
            competition: Competition name
            top_n: Number of top submissions to analyze

        Returns:
            Dictionary with submission analysis
        """
        try:
            submissions = self.competition_client.get_competition_submissions(
                competition
            )

            # Convert to DataFrame for analysis
            submissions_df = pd.DataFrame(submissions)

            analysis = {
                'total_submissions': len(submissions_df),
                'best_score': submissions_df['score'].max(),
                'average_score': submissions_df['score'].mean(),
                'score_progression': self._analyze_score_progression(submissions_df),
                'top_submissions': submissions_df.nlargest(top_n, 'score').to_dict('records')
            }

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing submission history: {str(e)}")
            raise

    def _analyze_score_progression(
        self,
        submissions_df: pd.DataFrame
    ) -> List[Dict]:
        """Analyze score progression over time"""
        try:
            submissions_df['timestamp'] = pd.to_datetime(submissions_df['timestamp'])
            submissions_df = submissions_df.sort_values('timestamp')

            return [
                {
                    'submission_number': i + 1,
                    'score': row['score'],
                    'timestamp': row['timestamp'].isoformat(),
                    'improvement': row['score'] - submissions_df.iloc[i-1]['score'] if i > 0 else 0
                }
                for i, row in submissions_df.iterrows()
            ]

        except Exception as e:
            logger.error(f"Error analyzing score progression: {str(e)}")
            raise

    @timer
    def get_competition_status(self, competition: str) -> Dict:
        """
        Get current competition status

        Args:
            competition: Competition name

        Returns:
            Dictionary with competition status
        """
        try:
            details = self.competition_client.get_competition_details(competition)
            leaderboard_analysis = self.track_leaderboard(competition, store_history=False)
            submission_analysis = self.analyze_submission_history(competition)

            status = {
                'competition': competition,
                'deadline': details.get('deadline'),
                'total_teams': details.get('totalTeams'),
                'current_rank': submission_analysis.get('best_rank'),
                'best_score': submission_analysis.get('best_score'),
                'submissions_remaining': details.get('maxSubmissionsPerDay') - submission_analysis.get('today_submissions', 0),
                'leaderboard_stats': leaderboard_analysis,
                'submission_stats': submission_analysis
            }

            return status

        except Exception as e:
            logger.error(f"Error getting competition status: {str(e)}")
            raise

def main():
    """Example usage of leaderboard operations"""
    try:
        # Initialize manager
        manager = LeaderboardManager()

        # Get competition status
        competition_name = "titanic"
        status = manager.get_competition_status(competition_name)

        print("\nCompetition Status:")
        print(f"Competition: {status['competition']}")
        print(f"Deadline: {status['deadline']}")
        print(f"Total Teams: {status['total_teams']}")
        print(f"Current Rank: {status['current_rank']}")
        print(f"Best Score: {status['best_score']}")
        print(f"Submissions Remaining Today: {status['submissions_remaining']}")

        print("\nLeaderboard Stats:")
        leaderboard_stats = status['leaderboard_stats']
        print(f"Total Entries: {leaderboard_stats['total_entries']}")
        print(f"Top Score: {leaderboard_stats['top_score']}")
        print(f"Median Score: {leaderboard_stats['median_score']}")

        print("\nSubmission History:")
        submission_stats = status['submission_stats']
        print(f"Total Submissions: {submission_stats['total_submissions']}")
        print(f"Best Score: {submission_stats['best_score']}")
        print(f"Average Score: {submission_stats['average_score']}")

    except Exception as e:
        print(f"Error in leaderboard operations: {str(e)}")

if __name__ == "__main__":
    main()
