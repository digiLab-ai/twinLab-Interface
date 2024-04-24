import pytest
from unittest.mock import patch

import twinlab as tl


@pytest.fixture
def mock_api_response_load_example_dataset():
    return 200, {
        "dataset": "Pack price [GBP],Number of biscuits per pack,Number of packs sold,Profit [GBP]\n2.3,20,2385,16695.0\n2.25,20,2553,16594.5\n2.0,18,2812,15747.2\n1.8,16,2647,13764.4\n1.9,16,2755,17081.0\n1.95,16,2642,17701.4\n2.2,18,2516,19121.6\n2.0,20,2886,11544.0\n2.05,20,2964,13338.0\n2.1,20,2839,14195.0\n1.8,20,3438,6876.0\n1.9,18,2887,13280.2\n"
    }


@patch("twinlab.api.load_example_dataset")
def test_load_example_dataset(
    mock_api_load_example_dataset_function,
    mock_api_response_load_example_dataset,
    dataframe_regression,
):
    # Mock request and response
    mock_api_load_example_dataset_function.return_value = (
        mock_api_response_load_example_dataset
    )

    # Call function
    df = tl.load_example_dataset("biscuits")

    # Test function
    dataframe_regression.check(df, basename="load_example_dataset_biscuits")
