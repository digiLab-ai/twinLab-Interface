# Standard imports
import io

# Third-party imports
import pandas as pd

# Project imports
import twinlab.api as api


def test_success_response(upload_setup, dataframe_regression):
    # NOTE: upload_setup is a necessary argument, even though not used, in order to run the fixture

    # Test the success response
    status, body = api.analyse_dataset("test_dataset", columns=["X,y"])

    # convert csv body to df
    df = pd.read_csv(io.StringIO(body["dataframe"]))

    # Check the response status code
    assert status == 200

    # Check the response body
    dataframe_regression.check(df)
