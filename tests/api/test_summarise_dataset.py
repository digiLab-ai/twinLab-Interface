# Standard imports
import io

# Third-party imports
import pandas as pd
import pytest

# Project imports
import twinlab.api as api


def test_success_response(upload_setup, dataframe_regression):
    # NOTE: upload_setup is a necessary argument, even though not used, in order to run the fixture

    # Test the success response
    status, body = api.summarise_dataset("test_dataset")

    # convert csv body to df
    df = pd.read_csv(io.StringIO(body["dataset_summary"]))

    # Check the response status code
    assert status == 200

    # Compare values of expected and actual dataframes
    dataframe_regression.check(df)


def test_fail_response():
    with pytest.raises(Exception) as exception:

        # Test the failure response
        _, _ = api.summarise_dataset("invalid_dataset_id")

        # Check the response status code
        assert exception.status == 500

        # Check the response body
        assert (
            "Unable to load data from your account"
            and "datasets/invalid_dataset_id/summary.csv" in str(exception.value)
        )
