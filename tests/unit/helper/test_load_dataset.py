import csv
import pytest
import pandas as pd
from unittest.mock import patch

import twinlab as tl


@pytest.fixture
def mock_response_csv_file():
    return pd.DataFrame([["Joe", 2], ["Joseph", 1]], columns=["Name", "Credits"])


@pytest.fixture
def mock_response_csv_file_path(tmp_path):
    # Create a temporary CSV file
    csv_data = [["Name", "Credits"], ["Joe", "2"], ["Joseph", "1"]]
    csv_file_path = tmp_path / "test.csv"
    # Write the csv_data to the csv_file_path
    with open(csv_file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)
    # Return the path to the temporary CSV file
    return csv_file_path


@patch("twinlab.helper.load_dataset")
def test_load_dataset(
    mock_load_dataset_function,
    mock_response_csv_file,
    mock_response_csv_file_path,
    dataframe_regression,
):
    # Mock request and response
    mock_load_dataset_function.return_value = mock_response_csv_file

    # Load dataset
    df = tl.load_dataset(filepath=str(mock_response_csv_file_path))

    # Check the results
    dataframe_regression.check(df, basename="load_dataset_data")
