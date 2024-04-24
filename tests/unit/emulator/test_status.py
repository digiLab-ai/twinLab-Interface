from unittest.mock import patch

import pytest
import twinlab as tl


@pytest.fixture
def mock_api_response():
    return 200, {
        "process_id": "8b0af49f-7767-4728-bf58-74450fad1915",
        "process_status": "Your job has finished and is on its way back to you.",
    }


@patch("twinlab.api.train_response_model")
def test_status(
    api_train_response_model,
    mock_api_response,
    data_regression,
):
    # Arrange
    emulator = tl.Emulator(id="test_emulator")
    api_train_response_model.return_value = mock_api_response

    # Act
    process_id = "8b0af49f-7767-4728-bf58-74450fad1915"
    status = emulator.status(process_id)

    # Assert
    data_regression.check(status)
