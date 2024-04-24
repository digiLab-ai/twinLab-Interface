# Project imports
import twinlab.api as api


def test_success_response(upload_setup):
    # NOTE: upload_setup is a necessary argument, even though not used, in order to run the fixture

    status, body = api.list_datasets()

    # Check the response status code,
    assert status == 200

    # Check if the datasets key exists in the response body
    assert "datasets" in body.keys()

    # Check if the value of datasets is a list
    assert type(body["datasets"]) == list

    # Check if the test_dataset is in the list of datasets
    assert "test_dataset" in body["datasets"]
