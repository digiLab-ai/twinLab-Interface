# Third-party imports
import pytest

# Project imports
import twinlab.api as api


def test_success_response(upload_with_url_setup):
    # NOTE: upload_with_url_setup is a necessary argument, even though not used, in order to run the fixture

    # Test the success response
    status, body = api.process_uploaded_dataset("test_dataset")

    # Assert the response status code
    assert status == 200

    # assert the response body
    assert body == {"message": "Dataset test_dataset was processed."}


def test_fail_response():

    with pytest.raises(Exception) as exception:

        # Test the failure response
        _, _ = api.process_uploaded_dataset("invalid_dataset_id")

    # Assert the exception response
    assert (
        "Unable to load data from your account"
        and "datasets/invalid_dataset_id/data.csv" in str(exception.value)
    )
