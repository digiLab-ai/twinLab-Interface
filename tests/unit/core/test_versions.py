import pytest
from unittest.mock import patch

import twinlab as tl


@pytest.fixture
def mock_response_get_versions():
    return 200, {
        "cloud": "2.1.0",
        "image": "twinlab-dev",
        "library": "1.6.0",
        "modal": "0.2.0",
    }


@patch("twinlab.api.get_versions")
def test_versions(
    mock_api_get_versions_function, mock_response_get_versions, data_regression
):
    # Mock request and response
    mock_api_get_versions_function.return_value = mock_response_get_versions
    # Call function
    version_info = tl.core.versions()
    # Check response
    data_regression.check(version_info, basename="test_versions")
