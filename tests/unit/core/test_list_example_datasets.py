import pytest
from unittest.mock import patch

import twinlab as tl


@pytest.fixture
def mock_api_response_list_example_datasets():
    return 200, {"datasets": ["example1", "example2", "example3"]}


@patch("twinlab.api.list_example_datasets")
def test_list_example_datasets(
    mock_api_list_example_datasets_function,
    mock_api_response_list_example_datasets,
):
    # Mock request and response
    mock_api_list_example_datasets_function.return_value = (
        mock_api_response_list_example_datasets
    )

    # Call function
    datasets = tl.list_example_datasets()

    # Check result (Not dictionary since the actual ouput is a list)
    assert datasets == ["example1", "example2", "example3"]
