import pytest
from unittest.mock import patch

import twinlab as tl


@pytest.fixture
def mock_api_response_list_datasets():
    return 200, {"datasets": ["dataset1", "dataset2", "dataset3"]}


@patch("twinlab.api.list_datasets")
def test_list_datasets(
    mock_api_list_datasets_function,
    mock_api_response_list_datasets,
):
    # Mock request and response
    mock_api_list_datasets_function.return_value = mock_api_response_list_datasets
    # Call function (from core because this function is also in client)
    datasets = tl.core.list_datasets()
    # Check result (Not dictionary since the actual ouput is a list)
    assert datasets == ["dataset1", "dataset2", "dataset3"]
