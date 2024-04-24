# Standard imports
import os

# Third-party imports
import dotenv
import pytest

# Project imports
import twinlab.api as api


def test_success_response():

    # test the success response
    status, body = api.get_user()

    # assert the response
    assert status == 200
    assert list(body.keys()) == ["username", "credits"]


def test_fail_response():

    # temporarily set .env variables
    TWINLAB_API_KEY = "invalid_api_key"
    os.environ["TWINLAB_API_KEY"] = TWINLAB_API_KEY

    # test the failure response
    with pytest.raises(Exception) as exception:
        _, _ = api.get_user()

    # assert the error message
    assert (
        str(exception.value)
        == f"Error Code: 400 - {{'message': 'Unable to find user with API key: `{TWINLAB_API_KEY}`'}}"
    )

    # reset .env variables
    TWINLAB_API_KEY = dotenv.dotenv_values(dotenv.find_dotenv())["TWINLAB_API_KEY"]
    os.environ["TWINLAB_API_KEY"] = TWINLAB_API_KEY
