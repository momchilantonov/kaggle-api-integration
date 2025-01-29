from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import json
import csv
from io import StringIO
import functools
import time
from typing import Callable, Any
from config.settings import setup_logger

logger = setup_logger('data_handlers', 'data_handlers.log')

class DataHandler:
    """Handler for data processing and manipulation"""

    @staticmethod
    def read_csv(
        file_path: Union[str, Path],
        **kwargs
    ) -> pd.DataFrame:
        """
        Read CSV file with error handling and logging

        Args:
            file_path: Path to CSV file
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            DataFrame containing the data
        """
        try:
            df = pd.read_csv(file_path, **kwargs)
            logger.info(f"Successfully read CSV file: {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {str(e)}")
            raise

    @staticmethod
    def write_csv(
        df: pd.DataFrame,
        file_path: Union[str, Path],
        index: bool = False,
        **kwargs
    ) -> None:
        """
        Write DataFrame to CSV with error handling

        Args:
            df: DataFrame to write
            file_path: Output file path
            index: Whether to write index
            **kwargs: Additional arguments for df.to_csv
        """
        try:
            df.to_csv(file_path, index=index, **kwargs)
            logger.info(f"Successfully wrote CSV file: {file_path}")
        except Exception as e:
            logger.error(f"Error writing CSV file {file_path}: {str(e)}")
            raise

    @staticmethod
    def validate_submission_format(
        df: pd.DataFrame,
        required_columns: List[str],
        column_types: Optional[Dict[str, type]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate competition submission format

        Args:
            df: Submission DataFrame
            required_columns: List of required column names
            column_types: Dictionary of column names and their expected types

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check required columns
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")

        # Check column types
        if column_types:
            for col, expected_type in column_types.items():
                if col in df.columns:
                    if not all(isinstance(x, expected_type) for x in df[col].dropna()):
                        errors.append(
                            f"Column {col} contains values not of type {expected_type}"
                        )

        return len(errors) == 0, errors

    @staticmethod
    def process_large_csv(
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        processing_func: callable,
        chunk_size: int = 10000,
        **kwargs
    ) -> None:
        """
        Process large CSV files in chunks

        Args:
            input_path: Input CSV file path
            output_path: Output CSV file path
            processing_func: Function to apply to each chunk
            chunk_size: Number of rows per chunk
            **kwargs: Additional arguments for read_csv
        """
        try:
            # Process first chunk to get headers
            first_chunk = True
            for chunk in pd.read_csv(input_path, chunksize=chunk_size, **kwargs):
                processed_chunk = processing_func(chunk)

                if first_chunk:
                    processed_chunk.to_csv(output_path, mode='w', index=False)
                    first_chunk = False
                else:
                    processed_chunk.to_csv(
                        output_path,
                        mode='a',
                        header=False,
                        index=False
                    )

            logger.info(
                f"Successfully processed large CSV from {input_path} to {output_path}"
            )
        except Exception as e:
            logger.error(f"Error processing large CSV: {str(e)}")
            raise

    @staticmethod
    def split_dataset(
        df: pd.DataFrame,
        train_size: float = 0.8,
        random_state: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train and test sets

        Args:
            df: Input DataFrame
            train_size: Proportion of data for training
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_df, test_df)
        """
        if not 0 < train_size < 1:
            raise ValueError("train_size must be between 0 and 1")

        if random_state is not None:
            np.random.seed(random_state)

        mask = np.random.rand(len(df)) < train_size

        return df[mask], df[~mask]

    @staticmethod
    def handle_missing_values(
        df: pd.DataFrame,
        strategy: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Handle missing values in DataFrame

        Args:
            df: Input DataFrame
            strategy: Dict mapping column names to strategies
                     ('mean', 'median', 'mode', 'drop', or a constant value)

        Returns:
            DataFrame with handled missing values
        """
        df_copy = df.copy()

        for column, method in strategy.items():
            if column not in df_copy.columns:
                logger.warning(f"Column {column} not found in DataFrame")
                continue

            if method == 'drop':
                df_copy = df_copy.dropna(subset=[column])
            elif method in ['mean', 'median', 'mode']:
                if method == 'mean':
                    value = df_copy[column].mean()
                elif method == 'median':
                    value = df_copy[column].median()
                else:  # mode
                    value = df_copy[column].mode()[0]
                df_copy[column].fillna(value, inplace=True)
            else:
                df_copy[column].fillna(method, inplace=True)

        return df_copy

    @staticmethod
    def calculate_basic_stats(df: pd.DataFrame) -> Dict:
        """
        Calculate basic statistics for numeric columns

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with basic statistics
        """
        stats = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            stats[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'missing': df[col].isnull().sum()
            }

        return stats

    @staticmethod
    def sample_dataset(
        df: pd.DataFrame,
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Sample rows from DataFrame

        Args:
            df: Input DataFrame
            n: Number of rows to sample
            frac: Fraction of rows to sample
            random_state: Random seed for reproducibility

        Returns:
            Sampled DataFrame
        """
        if random_state is not None:
            np.random.seed(random_state)

        if n is not None:
            return df.sample(n=n)
        elif frac is not None:
            return df.sample(frac=frac)
        else:
            raise ValueError("Must provide either n or frac")

    @staticmethod
    def convert_dtypes(
        df: pd.DataFrame,
        dtype_map: Dict[str, type]
    ) -> pd.DataFrame:
        """
        Convert column data types

        Args:
            df: Input DataFrame
            dtype_map: Dictionary mapping column names to types

        Returns:
            DataFrame with converted types
        """
        df_copy = df.copy()

        for column, dtype in dtype_map.items():
            if column not in df_copy.columns:
                logger.warning(f"Column {column} not found in DataFrame")
                continue

            try:
                df_copy[column] = df_copy[column].astype(dtype)
            except Exception as e:
                logger.error(f"Error converting {column} to {dtype}: {str(e)}")
                raise

        return df_copy



def timer(func: Callable) -> Callable:
    """Decorator for timing function execution"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"{func.__name__} completed in {duration:.2f} seconds")
        return result
    return wrapper
