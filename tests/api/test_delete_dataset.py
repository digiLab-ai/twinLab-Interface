# Third-party imports
import pytest

# Project imports
import twinlab.api as api


def test_success_response(upload_setup):
    # NOTE: upload_setup is a necessary argument, even though not used, in order to run the fixture

    # Test the success response
    status, body = api.delete_dataset("test_dataset")

    # Check the response status code
    assert status == 200

    # Check the response body
    assert body["message"] == "Dataset test_dataset deleted from the cloud."


def test_fail_response():

    with pytest.raises(Exception) as exception:

        dataset_id = "invalid_dataset"

        # Test the failure response
        _, _ = api.view_dataset(dataset_id)

        # Check the response status code
        assert exception.status == 500

        # Check the response body
        assert f"Dataset {dataset_id} not found on the cloud." in str(exception.value)
