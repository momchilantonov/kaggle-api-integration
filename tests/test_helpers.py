import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json
import os
from typing import Dict

from src.handlers.data_handlers import DataHandler

@pytest.fixture
def sample_df():
    """Fixture to create a sample DataFrame"""
    return pd.DataFrame({
        'id': range(1, 6),
        'value': [10.0, 20.0, None, 40.0, 50.0],
        'category': ['A', 'B', 'A', None, 'B']
    })

@pytest.fixture
def sample_csv_path(tmp_path, sample_df):
    """Fixture to create a sample CSV file"""
    file_path = tmp_path / "test.csv"
    sample_df.to_csv(file_path, index=False)
    return file_path

def test_read_csv(sample_csv_path):
    """Test reading CSV file"""
    df = DataHandler.read_csv(sample_csv_path)
    assert len(df) == 5
    assert list(df.columns) == ['id', 'value', 'category']

def test_read_csv_invalid_path():
    """Test reading CSV from invalid path"""
    with pytest.raises(Exception):
        DataHandler.read_csv("nonexistent.csv")

def test_write_csv(tmp_path, sample_df):
    """Test writing DataFrame to CSV"""
    output_path = tmp_path / "output.csv"

    # Write the DataFrame
    DataHandler.write_csv(sample_df, output_path)
    assert output_path.exists()

    # Read back and verify
    df_read = pd.read_csv(output_path)

    # Verify basic properties
    assert df_read.shape == sample_df.shape
    assert all(df_read.columns == sample_df.columns)

    # Compare non-null values
    for column in sample_df.columns:
        # Get mask for non-null values in both DataFrames
        mask = sample_df[column].notna() & df_read[column].notna()
        # Compare values where both are non-null
        pd.testing.assert_series_equal(
            sample_df[column][mask],
            df_read[column][mask],
            check_dtype=False
        )

def test_validate_submission_format(sample_df):
    """Test submission format validation"""
    # Test with valid format
    is_valid, errors = DataHandler.validate_submission_format(
        sample_df,
        ['id', 'value'],
        {'id': int, 'value': float}
    )
    assert is_valid
    assert not errors

    # Test with missing column
    is_valid, errors = DataHandler.validate_submission_format(
        sample_df,
        ['id', 'nonexistent'],
        {'id': int}
    )
    assert not is_valid
    assert len(errors) == 1

def test_process_large_csv(tmp_path, sample_df):
    """Test processing large CSV in chunks"""
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"

    # Create a larger DataFrame
    large_df = pd.concat([sample_df] * 3, ignore_index=True)
    large_df.to_csv(input_path, index=False)

    # Process function that doubles the 'value' column
    def process_chunk(chunk):
        chunk['value'] = chunk['value'] * 2
        return chunk

    DataHandler.process_large_csv(input_path, output_path, process_chunk, chunk_size=2)

    # Verify results
    result_df = pd.read_csv(output_path)
    assert len(result_df) == len(large_df)
    assert all(result_df['value'].dropna() == large_df['value'].dropna() * 2)

def test_split_dataset(sample_df):
    """Test dataset splitting"""
    train_df, test_df = DataHandler.split_dataset(
        sample_df,
        train_size=0.6,
        random_state=42
    )

    assert len(train_df) + len(test_df) == len(sample_df)
    assert round(len(train_df) / len(sample_df), 1) == 0.6

def test_split_dataset_invalid_size():
    """Test dataset splitting with invalid size"""
    with pytest.raises(ValueError):
        DataHandler.split_dataset(pd.DataFrame(), train_size=1.5)

def test_handle_missing_values(sample_df):
    """Test handling missing values"""
    strategy = {
        'value': 'mean',
        'category': 'mode'
    }

    # Store original mean before filling
    original_mean = sample_df['value'].mean()

    result_df = DataHandler.handle_missing_values(sample_df, strategy)

    # Verify nulls are filled
    assert not result_df['value'].isnull().any()
    assert not result_df['category'].isnull().any()

    # Verify the mean value was used to fill
    null_mask = sample_df['value'].isnull()
    assert result_df.loc[null_mask, 'value'].iloc[0] == original_mean

def test_calculate_basic_stats(sample_df):
    """Test calculating basic statistics"""
    stats = DataHandler.calculate_basic_stats(sample_df)

    assert 'id' in stats
    assert 'value' in stats
    assert 'mean' in stats['value']
    assert 'median' in stats['value']
    assert 'std' in stats['value']
    assert stats['value']['missing'] == 1

def test_sample_dataset(sample_df):
    """Test dataset sampling"""
    # Test with n
    sampled_n = DataHandler.sample_dataset(sample_df, n=3, random_state=42)
    assert len(sampled_n) == 3

    # Test with frac
    sampled_frac = DataHandler.sample_dataset(sample_df, frac=0.6, random_state=42)
    assert len(sampled_frac) == 3  # 60% of 5 rounded down

    # Test invalid params
    with pytest.raises(ValueError):
        DataHandler.sample_dataset(sample_df)

def test_convert_dtypes(sample_df):
    """Test converting column data types"""
    # Test successful conversion
    result_df = DataHandler.convert_dtypes(sample_df, {'id': str})
    assert result_df['id'].dtype == np.dtype('O')

    # Test failed conversion
    with pytest.raises(Exception):
        DataHandler.convert_dtypes(sample_df, {'value': int})

def test_handle_missing_values_with_constant(sample_df):
    """Test handling missing values with constant values"""
    strategy = {
        'value': 999,
        'category': 'X'
    }

    result_df = DataHandler.handle_missing_values(sample_df, strategy)

    assert result_df['value'].fillna(0).eq(999).any()
    assert result_df['category'].fillna('').eq('X').any()

def test_handle_missing_values_with_drop(sample_df):
    """Test handling missing values with drop strategy"""
    strategy = {
        'value': 'drop',
        'category': 'drop'
    }

    result_df = DataHandler.handle_missing_values(sample_df, strategy)

    assert len(result_df) < len(sample_df)
    assert not result_df['value'].isnull().any()
    assert not result_df['category'].isnull().any()

def test_process_large_csv_with_error(tmp_path, sample_df):
    """Test process_large_csv with error in processing function"""
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    sample_df.to_csv(input_path, index=False)

    def error_process(chunk):
        raise ValueError("Processing error")

    with pytest.raises(ValueError):
        DataHandler.process_large_csv(input_path, output_path, error_process)

def test_validate_submission_format_with_invalid_types(sample_df):
    """Test submission format validation with invalid types"""
    is_valid, errors = DataHandler.validate_submission_format(
        sample_df,
        ['id', 'value'],
        {'value': int}  # value column contains floats
    )
    assert not is_valid
    assert len(errors) == 1
    assert "not of type" in errors[0]
