from unittest.mock import patch

import pytest
import twinlab as tl


@pytest.fixture
def mock_analyse_dataset_response():
    return 200, {
        "dataframe": ",Number of Dimensions,Cumulative Variance\n0,0,0.0\n1,1,0.9257412035415248\n2,2,1.0000000000000004\n"
    }


@patch("twinlab.api.analyse_dataset")
def test_analyse_variance(
    mock_analyse_dataset, mock_analyse_dataset_response, dataframe_regression
):
    # Arrange
    dataset = tl.Dataset("dataset")
    mock_analyse_dataset.return_value = mock_analyse_dataset_response

    # Act
    df = dataset.analyse_variance(["X", "y"])

    # Assert
    dataframe_regression.check(df, basename="analyse_input_variance")
