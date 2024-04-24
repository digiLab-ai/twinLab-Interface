# Third-party imports
import pytest

# Project imports
import twinlab.api as api


def test_success_response(upload_setup):
    # NOTE: upload_setup is a necessary argument, even though not used, in order to run the fixture

    status, body = api.train_request_model(
        "test_model",
        '{"dataset_id": "test_dataset", "inputs": ["A"], "outputs": ["B"], "train_test_ratio": 1}',
        "cpu",
    )

    assert status == 200
    assert body["message"] == "Model test_model has begun training."
    assert "process_id" in body


def test_fail_response_with_invalid_dataset():

    with pytest.raises(Exception) as exception:

        # Test the failure response
        _, _ = api.train_request_model(
            "test_model",
            '{"dataset_id": "invalid_dataset", "inputs": ["A"], "outputs": ["B"], "train_test_ratio": 1}',
            "cpu",
        )
    # Assert the exception response
    assert (
        "Error Code: 400 - {'message': 'The dataset you are trying to train on does not exist on your twinLab cloud account. Please ensure you have uploaded your dataset.'}"
        == str(exception.value)
    )
