# Third-party imports
import pytest

# Project imports
import twinlab.api as api


def test_success_response(training_setup):
    # NOTE: training_setup is a necessary argument, even though not used, in order to run the fixture

    # Test the success response
    status, body = api.delete_model("test_model")

    # Check the response status code
    assert status == 200

    # Check the response body
    assert body["message"] == f"Emulator test_model deleted from the cloud."


def test_fail_response():

    with pytest.raises(Exception) as exception:

        model_id = "invalid_model"

        # Test the failure response
        _, _ = api.delete_model(model_id)

        # Check the response status code
        assert exception.status == 500

        # Check the response body
        assert f"Emulator {model_id} not found on the cloud." in str(exception.value)
