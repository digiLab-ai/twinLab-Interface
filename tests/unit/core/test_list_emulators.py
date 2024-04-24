import pytest
from unittest.mock import patch

import twinlab as tl


@pytest.fixture
def mock_api_response_list_emulators():
    return 200, {
        "model_information": [
            {
                "model": "emulator1",
                "process_id": "585e6de9-06e0-443a-b213-838c83144611",
                "status": "success",
            },
            {
                "model": "emulator2",
                "process_id": "20b454dd-6b51-40bc-94be-445fd8e7c3f7",
                "status": "success",
            },
            {
                "model": "emulator3",
                "process_id": "6fbffab5-71cd-4d56-9272-e3f945d42d3c",
                "status": "success",
            },
        ],
        "models": ["emulator1", "emulator2", "emulator3"],
    }


@patch("twinlab.api.list_models")
def test_list_emulators(
    mock_api_list_emulators_function,
    mock_api_response_list_emulators,
    data_regression,
):
    # Mock request and response
    mock_api_list_emulators_function.return_value = mock_api_response_list_emulators
    # Call function (from core because this function is also in client)
    emulators = tl.core.list_emulators()
    # Check result
    data_regression.check(emulators, basename="list_emulators")
