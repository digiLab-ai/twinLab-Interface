import json
import pytest
from unittest.mock import patch

import twinlab as tl


@pytest.fixture
def mock_response_params_file():
    return {"param1": "value1", "param2": "value2"}


@pytest.fixture
def mock_response_params_file_path(tmp_path):
    # Create a temporary JSON file
    json_file_path = tmp_path / "test.json"
    # Write the json_data to the json_file_path
    with open(json_file_path, "w") as file:
        json.dump({"param1": "value1", "param2": "value2"}, file)
    # Return the path to the temporary JSON file
    return json_file_path


@patch("twinlab.helper.load_params")
def test_load_params(
    mock_load_params_function,
    mock_response_params_file,
    mock_response_params_file_path,
    data_regression,
):
    # Mock request and response
    mock_load_params_function.return_value = mock_response_params_file

    # Load params
    params = tl.load_params(filepath=str(mock_response_params_file_path))

    # Check the results
    data_regression.check(params, basename="load_params_data")
