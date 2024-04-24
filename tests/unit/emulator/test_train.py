from unittest.mock import patch

import pytest
import twinlab as tl


@pytest.fixture
def mock_train_request_model_success():
    return 202, {
        "message": "Model test has begun training.",
        "process_id": "02f6c171-0743-4ff6-a2fc-c1dca1fc8f1e",
    }


@pytest.fixture
def mock_train_response_model_success():
    return 200, {
        "process_id": "69f25ac8-ab73-4067-a625-8f0183f1840d",
        "process_status": "done_Modal",
    }


@patch("twinlab.api.train_response_model")
@patch("twinlab.api.train_request_model")
def test_train(
    mock_train_request_model,
    train_response_model,
    mock_train_request_model_success,
    mock_train_response_model_success,
    capsys,
):
    # Arange
    emulator = tl.Emulator(id="test_emulator")
    dataset = tl.Dataset("test")
    inputs = ["X"]
    outputs = ["y"]
    mock_train_request_model.return_value = mock_train_request_model_success
    train_response_model.return_value = mock_train_response_model_success
    process_id = mock_train_request_model_success[1]["process_id"]
    expected_message = (
        f"Training of emulator {emulator.id} with process ID {process_id} is complete!"
    )

    # Act
    emulator.train(dataset, inputs, outputs, verbose=True)

    # Assert
    captured = capsys.readouterr()
    assert expected_message in captured.out


# TODO: More??
