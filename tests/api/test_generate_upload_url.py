# Project imports
import twinlab.api as api


def test_success_response():

    # test the success response
    status, body = api.generate_upload_url("test_dataset")

    # assert the response status code
    assert status == 200

    # assert a url to the quarantine bucket is created with the correct path
    assert "https://twinlab-quarantine" and "datasets/test_dataset/" in body["url"]
