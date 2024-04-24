from unittest.mock import patch

import pytest
import twinlab as tl

mock_process = {
    "process_id": "69f25ac8-ab73-4067-a625-8f0183f1840d",
    "method": "predict",
    "run-time": "0:00:06",
    "start_time": "2024-03-18 15:13:47",
    "status": "success",
}

mock_response_body = {"processes": [mock_process]}


@pytest.fixture
def mock_api_response_list_processes_model_response():
    return 200, {
        "processes": [
            {
                "process_id": "69f25ac8-ab73-4067-a625-8f0183f1840d",
                "method": "predict",
                "run-time": "0:00:06",
                "start_time": "2024-03-18 15:13:47",
                "status": "success",
            }
        ]
    }


@patch("twinlab.api.list_processes_model")
def test_list_processes(
    mock_api_response_list_processes_model,
    mock_api_response_list_processes_model_response,
    data_regression,
):
    # Arrange
    emulator = tl.Emulator(id="test_emulator")
    mock_api_response_list_processes_model.return_value = (
        mock_api_response_list_processes_model_response
    )

    # Act
    processes = emulator.list_processes()

    # Assert
    data_regression.check(processes)
