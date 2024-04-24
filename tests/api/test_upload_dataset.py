# Third-party imports
import pandas as pd
import pytest

# Project imports
import twinlab.api as api


def test_success_response():

    # Create a dataframe as a csv string
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    csv_string = df.to_csv(index=False)

    # Test the success response
    status, body = api.upload_dataset("test_dataset", csv_string)

    # Assert the response status code
    assert status == 200

    # Assert the response body
    assert body == {"message": "Dataset test_dataset was uploaded."}


def test_fail_response_minimum_column_number():

    with pytest.raises(Exception) as exception:

        # Test the failure response
        _, _ = api.upload_dataset("test_dataset", "onecolumn")

    # Assert the response body
    assert (
        str(exception.value)
        == "Error Code: 500 - {'message': 'Dataset must have at least one column.'}"
    )


def test_fail_response_cannot_parse():

    with pytest.raises(Exception) as exception:

        # Test the failure response
        _, _ = api.upload_dataset("test_dataset", "\n")

    # Assert the response body
    assert (
        str(exception.value)
        == "Error Code: 500 - {'message': 'Could not parse the input into a dataframe.'}"
    )
