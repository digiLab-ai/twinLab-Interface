import pytest
from unittest.mock import patch

import twinlab as tl


@pytest.fixture
def mock_response_user_information():
    return 200, {"credits": 0, "username": "your_username"}


@patch("twinlab.api.get_user")
def test_user_information(
    mock_api_get_user_function, mock_response_user_information, data_regression
):
    # Mock request and response
    mock_api_get_user_function.return_value = mock_response_user_information
    # Call function
    user_info = tl.user_information()
    # Check response
    data_regression.check(user_info, basename="test_user_information")
