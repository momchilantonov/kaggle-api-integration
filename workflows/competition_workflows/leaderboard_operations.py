from pathlib import Path
from typing import Dict, Optional, Union, List
import logging
from datetime import datetime
import pandas as pd
import json

from src.api.competitions import CompetitionClient
from src.utils.path_manager import PathManager
from src.utils.error_handlers import handle_api_errors
from src.handlers.data_handlers import DataHandler

logger = logging.getLogger(__name__)

class LeaderboardManager:
    """Manages competition leaderboard workflows"""

    def __init__(self):
        self.competition_client = CompetitionClient()
        self.data_handler = DataHandler()
        self.path_manager = PathManager()
        # Ensure required directories exist
        self.path_manager.ensure_directories()

    @handle_api_errors
    def track_leaderboard(
        self,
        competition: str,
        store_history: bool = True
    ) -> Dict:
        """Download and analyze current leaderboard"""
        try:
            # Download current leaderboard
            leaderboard_path = self.competition_client.download_leaderboard(
                competition,
                path=self.path_manager.get_path('competitions', 'leaderboards')
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
                'top_score': leaderboard_df.iloc[0]['Score'] if not leaderboard_df.empty else None,
                'median_score': leaderboard_df['Score'].median() if not leaderboard_df.empty else None,
                'score_std': leaderboard_df['Score'].std() if not leaderboard_df.empty else None,
                'timestamp': datetime.now().isoformat()
            }

            # Add percentile analysis
            if not leaderboard_df.empty:
                percentiles = [25, 50, 75, 90, 95, 99]
                analysis['percentiles'] = {
                    f'p{p}': leaderboard_df['Score'].quantile(p/100)
                    for p in percentiles
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
            history_dir = (
                self.path_manager.get_path('competitions', 'leaderboards') /
                'history' /
                competition
            )
            history_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            history_path = history_dir / f"leaderboard_{timestamp}.csv"

            leaderboard_df.to_csv(history_path, index=False)
            logger.info(f"Stored leaderboard history: {history_path}")

            # Create a summary of changes
            self._create_change_summary(competition, leaderboard_df, history_dir)

            return history_path

        except Exception as e:
            logger.error(f"Error storing leaderboard history: {str(e)}")
            raise

    def _create_change_summary(
        self,
        competition: str,
        current_leaderboard: pd.DataFrame,
        history_dir: Path
    ) -> None:
        """Create summary of leaderboard changes"""
        try:
            # Get previous leaderboard if exists
            previous_files = sorted(history_dir.glob('leaderboard_*.csv'))
            if len(previous_files) > 1:
                previous_leaderboard = pd.read_csv(previous_files[-2])

                # Compare leaderboards
                changes = {
                    'new_entries': len(current_leaderboard) - len(previous_leaderboard),
                    'top_score_change': (
                        current_leaderboard.iloc[0]['Score'] -
                        previous_leaderboard.iloc[0]['Score']
                        if not current_leaderboard.empty and not previous_leaderboard.empty
                        else 0
                    ),
                    'timestamp': datetime.now().isoformat()
                }

                # Save changes summary
                summary_path = history_dir / 'changes_summary.json'
                if summary_path.exists():
                    with open(summary_path, 'r') as f:
                        summary_history = json.load(f)
                else:
                    summary_history = []

                summary_history.append(changes)
                with open(summary_path, 'w') as f:
                    json.dump(summary_history, f, indent=2)

        except Exception as e:
            logger.error(f"Error creating change summary: {str(e)}")
            # Don't raise exception as this is a non-critical operation

    @handle_api_errors
    def get_leaderboard_trends(
        self,
        competition: str,
        days: int = 7
    ) -> Dict:
        """Analyze leaderboard trends over time"""
        try:
            history_dir = (
                self.path_manager.get_path('competitions', 'leaderboards') /
                'history' /
                competition
            )

            if not history_dir.exists():
                return {'error': 'No historical data available'}

            # Get all leaderboard files within the time range
            cutoff_date = datetime.now() - pd.Timedelta(days=days)
            history_files = []

            for file_path in history_dir.glob('leaderboard_*.csv'):
                file_date = datetime.strptime(
                    file_path.stem.split('_')[1],
                    "%Y%m%d_%H%M%S"
                )
                if file_date >= cutoff_date:
                    history_files.append(file_path)

            if not history_files:
                return {'error': f'No data available for the last {days} days'}

            # Analyze trends
            trends = {
                'dates': [],
                'top_scores': [],
                'median_scores': [],
                'total_participants': []
            }

            for file_path in sorted(history_files):
                df = pd.read_csv(file_path)
                file_date = datetime.strptime(
                    file_path.stem.split('_')[1],
                    "%Y%m%d_%H%M%S"
                ).isoformat()

                trends['dates'].append(file_date)
                trends['top_scores'].append(df.iloc[0]['Score'] if not df.empty else None)
                trends['median_scores'].append(df['Score'].median() if not df.empty else None)
                trends['total_participants'].append(len(df))

            return trends

        except Exception as e:
            logger.error(f"Error analyzing leaderboard trends: {str(e)}")
            raise

if __name__ == '__main__':
    # Example usage
    manager = LeaderboardManager()

    try:
        # Track current leaderboard
        analysis = manager.track_leaderboard("titanic", store_history=True)
        print(f"Current leaderboard analysis: {analysis}")

        # Get trends
        trends = manager.get_leaderboard_trends("titanic", days=7)
        print(f"Leaderboard trends: {trends}")

    except Exception as e:
        print(f"Error: {str(e)}")
