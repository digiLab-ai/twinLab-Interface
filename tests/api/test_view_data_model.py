# Standard imports
import io

# Third-party imports
import pandas as pd
import pytest

# Project imports
import twinlab.api as api


def test_training_success_response(training_setup, dataframe_regression):
    # NOTE: training_setup is a necessary argument, even though not used, in order to run the fixture

    status, body = api.view_data_model("test_model", "train")
    df = pd.read_csv(io.StringIO(body["training_data"]))

    assert status == 200

    dataframe_regression.check(df)


def test_test_success_response(dataframe_regression):

    status, body = api.view_data_model("test_model", "test")
    df = pd.read_csv(io.StringIO(body["test_data"]))

    assert status == 200

    dataframe_regression.check(df)


def test_train_fail_response():

    with pytest.raises(Exception) as exception:

        # Test the failure response
        _, _ = api.view_data_model("invalid_model", "train")

    # Assert the exception response
    assert (
        "Error Code: 400 - {'message': 'Unable to load data from your account:"
        and "models/invalid_model/training_data.csv" in str(exception.value)
    )


def test_test_fail_response():

    with pytest.raises(Exception) as exception:

        # Test the failure response
        _, _ = api.view_data_model("invalid_model", "test")

    # Assert the exception response
    assert (
        "Error Code: 400 - {'message': 'Unable to load data from your account:"
        and "models/invalid_model/test_data.csv" in str(exception.value)
    )
