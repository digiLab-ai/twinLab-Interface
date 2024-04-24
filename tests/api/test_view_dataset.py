# Third-party imports
import pytest

# Project imports
import twinlab.api as api


def test_success_response(upload_setup):
    # NOTE: upload_setup is a necessary argument, even though not used, in order to run the fixture

    # Test the success response
    status, body = api.view_dataset("test_dataset")

    # Check the response status code
    assert status == 200

    # Check the response body
    assert body["dataset"] == "X,y\n1,4\n2,5\n3,6\n"


def test_fail_response():

    with pytest.raises(Exception) as exception:

        # Test the failure response
        _, _ = api.view_dataset("invalid_dataset_id")

        # Check the response status code
        assert exception.status == 500

        # Check the response body
        assert (
            "Unable to load data from your account"
            and "datasets/invalid_dataset_id/data.csv" in str(exception.value)
        )
