# Project imports
import twinlab.api as api


def test_success_response(training_setup):
    # NOTE: training_setup is a necessary argument, even though not used, in order to run the fixture

    # Create a model
    status, body = api.list_models()

    # Check for successful response and the presence of the models key in the response body
    assert status == 200

    # Check the presence of the models key in the response body
    assert "models" in body.keys()

    # Check if the value of models is a list
    assert type(body["models"]) == list

    # Check if the test_model is in the list of models
    assert "test_model" in body["models"]
