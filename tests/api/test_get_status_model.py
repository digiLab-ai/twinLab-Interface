# Third-party imports
import pytest

# Project imports
import twinlab.api as api


def test_success_response(training_setup):
    # NOTE: training_setup is a necessary argument, even though not used, in order to run the fixture

    status, body = api.get_status_model(
        "test_model",
    )

    assert status == 200
    assert body["job_complete"] == True
    assert body["message"] == "Training job is complete."


def test_fail_response():

    with pytest.raises(Exception) as exception:

        # Test the failure response
        _, _ = api.get_status_model(
            "non_existent_model",
        )

    assert "{'message': 'Unable to load data from your account:" in str(exception.value)
