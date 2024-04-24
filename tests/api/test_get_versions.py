# Project imports
import twinlab.api as api


def test_success_response():

    # test the success response
    status, body = api.get_versions()

    # assert the response
    assert status == 200
    # TODO Is there a way to load the stack versions and assert they are the correct?
    assert list(body.keys()) == ["cloud", "modal", "library", "image"]
