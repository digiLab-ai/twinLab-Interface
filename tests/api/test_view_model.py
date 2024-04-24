# Third-party imports
import pytest

# Project imports
import twinlab.api as api


def test_success_response(training_setup):
    # NOTE: training_setup is a necessary argument, even though not used, in order to run the fixture

    # Create a model
    status, body = api.view_model("test_model")
    body.pop("modal_handle")

    # Check for successful response and the presence of the models key in the response body
    assert status == 200

    assert body == {
        "model_id": "test_model",
        "dataset_id": "test_dataset",
        "inputs": ["X"],
        "outputs": ["y"],
        "train_test_ratio": 0.8,
    }


def test_fail_response():

    # Create an invalid model ID
    model_id = "invalid_model_id"

    with pytest.raises(Exception) as exception:

        # Test the failure response
        _, _ = api.view_model(model_id)

        # Check the response status code
        assert exception.status == 500

        # Check the response body
        assert (
            "Unable to load data from your account"
            and "models/invalid_model_id/meta.json" in str(exception.value)
        )
