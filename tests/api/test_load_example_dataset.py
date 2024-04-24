# Standard imports
import io

# Third-party imports
import pandas as pd
import pytest

# Project imports
import twinlab.api as api


def test_success_response_quickstart(dataframe_regression):

    # Test the success response
    status, body = api.load_example_dataset("quickstart")

    # convert csv body to df
    df = pd.read_csv(io.StringIO(body["dataset"]))

    # Check the response status code
    assert status == 200

    dataframe_regression.check(df)


def test_success_response_biscuits(dataframe_regression):

    # Test the success response
    status, body = api.load_example_dataset("biscuits")

    # convert csv body to df
    df = pd.read_csv(io.StringIO(body["dataset"]))

    # Check the response status code
    assert status == 200

    dataframe_regression.check(df)


def test_success_response_gardening(dataframe_regression):

    # Test the success response
    status, body = api.load_example_dataset("gardening")

    # convert csv body to df
    df = pd.read_csv(io.StringIO(body["dataset"]))

    # Check the response status code
    assert status == 200

    dataframe_regression.check(df)


def test_success_response_tritium_desorption(dataframe_regression):

    # Test the success response
    status, body = api.load_example_dataset("tritium-desorption")

    # convert csv body to df
    df = pd.read_csv(io.StringIO(body["dataset"]))

    # Check the response status code
    assert status == 200

    dataframe_regression.check(df)


def test_fail_response():

    with pytest.raises(Exception) as exception:

        # Test the failure response
        _, _ = api.load_example_dataset("invalid_example_dataset")

    # Assert the response body
    assert (
        "Error Code: 400 - {'message': \"Dataset invalid_example_dataset does not exist."
        in str(exception.value)
    )
