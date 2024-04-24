# Package imports
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

import twinlab as tl
from twinlab import utils


class TestCoerceParams:
    """
    Unit tests for tl.utils.coerce_params_dict.
    """

    def test_coerce_error(self):
        """
        Test for twinlab.utils.coerce_params_dict()

        Requirements
        ------------
        method raises the correct error for deprecated column name.

        """
        # Setup
        params = {
            "dataset_id": "my_dataset",
            "inputs": ["X"],
            "outputs": ["y"],
            "train_test_split": 1.0,
        }

        with pytest.raises(TypeError) as excinfo:
            tl.utils.coerce_params_dict(params)

        # Test error message
        assert (
            str(excinfo.value)
            == "train_test_split is deprecated. Use train_test_ratio instead."
        )

    def test_coerce_add(self):
        """
        Test for twinlab.utils.coerce_params_dict()

        Requirements
        ------------
        method adds a train_test_ratio column to the parameter dictionary

        """
        # Setup
        params_in = {
            "dataset_id": "my_dataset",
            "inputs": ["X"],
            "outputs": ["y"],
        }
        params_out = {
            "dataset_id": "my_dataset",
            "inputs": ["X"],
            "outputs": ["y"],
            "train_test_ratio": 1.0,
        }

        output = tl.utils.coerce_params_dict(params_in)

        assert output == params_out


class TestCheckDataset:
    """
    Unit tests for tl.utils.check_dataset
    """

    def test_check_parse(self):
        """
        Test for twinlab.utils.check_dataset()

        Requirements
        ------------
        method raises the correct error when an un-parsable string is input

        """
        # Setup
        df = """
            Column1,Column2,Column3
            1,2,3
            4,,6
            7,8,9,10
            """

        with pytest.raises(TypeError) as excinfo:
            utils.check_dataset(df)

        # Test error message
        assert str(excinfo.value) == "Could not parse the input into a dataframe."
        # assert excinfo.type == Warning

    def test_check_duplicate(self):
        """
        Test for twinlab.utils.check_dataset()

        Requirements
        ------------
        method raises the correct error when there are duplicate column names

        """
        # Setup
        string = """Name,Age,City,Name
                    1, 30, 2 , 1
                    2, 25, 4 , 11
                    3, 32, 9, 111
                    """

        with pytest.raises(TypeError) as excinfo:
            utils.check_dataset(string)

        # Test error message
        assert str(excinfo.value) == "Dataset must contain no duplicate column names."

    def test_check_val(self):
        """
        Test for twinlab.utils.check_dataset()

        Requirements
        ------------
        method raises the correct error when the dataset contains a non-numerical value

        """
        # Setup
        string = """Name,Age,City,exp
                    1, 30, 2 , 1
                    2, 25, 4 , 11
                    3, 32, test, 111
                    """

        with pytest.raises(Warning) as excinfo:
            utils.check_dataset(string)

        # Test error message
        assert str(excinfo.value) == "Dataset contains non-numerical values."

    def test_check_shape_col(self):
        """
        Test for twinlab.utils.check_dataset()

        Requirements
        ------------
        method raises the correct error when the dataset contains <1 column

        """
        # Setup
        data = "1,2,3"

        with pytest.raises(TypeError) as excinfo:
            utils.check_dataset(data)

        # Test error message
        assert str(excinfo.value) == "Dataset must have at least one column."


@patch("requests.put")
def test_upload_file_to_url(mock_requests_put):
    """
    Test for twinlab.utils.upload_file_to_presigned_url()

    Requirements
    ------------
    method calls requests.put

    """
    # Setup
    # Create a temporary file and file_path
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_file.write(
            """
            1,2,3.5,4.8
            5.2,6,7.0,8.1
            9,10,11.2,12
            13.3,14,15,16.5
            """
        )
        temp_file_path = temp_file.name

    mock_response = MagicMock()
    mock_response.status_code = 200

    tl.utils.upload_file_to_presigned_url(
        temp_file_path, "https://example.com/api/data", verbose=True, check=True
    )

    mock_requests_put.assert_called_once()


@patch("requests.put")
def test_upload_file_to_url_fail(mock_put, capsys):
    """
    Test for twinlab.utils.upload_file_to_presigned_url()

    Requirements
    ------------
    method produces the correct error message when the requests.put fails

    """
    # Setup
    mock_put.return_value = MagicMock(
        spec=requests.models.Response,
        status_code=404,
        text="The requested resource could not be found on the server.",
    )

    # Create a temporary file and file_path
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_file.write(
            """
            1,2,3.5,4.8
            5.2,6,7.0,8.1
            9,10,11.2,12
            13.3,14,15,16.5
            """
        )
        temp_file_path = temp_file.name

    tl.utils.upload_file_to_presigned_url(
        temp_file_path, "https://example.com/api/data", verbose=True, check=True
    )

    captured = capsys.readouterr()

    assert "File upload failed" in captured.out
    assert "404" in captured.out
    assert "The requested resource could not be found on the server." in captured.out


@patch("requests.put")
def test_upload_file_to_url_pass(capsys):
    """
    Test for twinlab.utils.upload_file_to_presigned_url()

    Requirements
    ------------
    method passes with correct status code and prints the correct message

    """
    # Setup
    # Create a temporary file and file_path
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_file.write(
            """
                1,2,3.5,4.8
                5.2,6,7.0,8.1
                9,10,11.2,12
                13.3,14,15,16.5
                """
        )
        temp_file_path = temp_file.name

    mock_response = MagicMock()
    mock_response.status_code = 200

    tl.utils.upload_file_to_presigned_url(
        temp_file_path, "https://example.com/api/data", verbose=True, check=False
    )

    captured = capsys.readouterr()

    assert captured.out.startswith(f"File {temp_file_path} is uploading.")


@patch("requests.put")
def test_upload_df_to_url(mock_requests_put):
    """
    Test for twinlab.utils.upload_dataframe_to_presigned_url()

    Requirements
    ------------
    method calls requests.put

    """
    # Setup
    # Create a temporary file path so the function can be called.
    data = {
        "Column1": [1, 2, 3, 4],
        "Column2": [5, 6, 7, 8],
        "Column3": [9, 10, 11, 12],
    }

    df = pd.DataFrame(data)

    mock_response = MagicMock()
    mock_response.status_code = 200

    tl.utils.upload_dataframe_to_presigned_url(
        df, "https://example.com/api/data", verbose=True
    )

    mock_requests_put.assert_called_once()


@patch("requests.put")
def test_upload_df_to_url_fail(mock_put, capsys):
    """
    Test for twinlab.utils.upload_dataframe_to_presigned_url()

    Requirements
    ------------
    method produces the correct error message when the requests.put fails

    """
    # Setup
    mock_put.return_value = MagicMock(
        spec=requests.models.Response,
        status_code=404,
        text="The requested resource could not be found on the server.",
    )

    data = {
        "Column1": [1, 2, 3, 4],
        "Column2": [5, 6, 7, 8],
        "Column3": [9, 10, 11, 12],
    }

    df = pd.DataFrame(data)

    tl.utils.upload_dataframe_to_presigned_url(
        df, "https://example.com/api/data", verbose=True, check=True
    )

    captured = capsys.readouterr()

    assert "Dataframe upload failed" in captured.out
    assert "404" in captured.out
    assert "The requested resource could not be found on the server." in captured.out


@patch("requests.put")
def test_upload_df_to_url_pass(capsys):
    """
    Test for twinlab.utils.upload_dataframe_to_presigned_url()

    Requirements
    ------------
    method passes with correct status code and prints the correct message

    """
    # Setup
    data = {
        "Column1": [1, 2, 3, 4],
        "Column2": [5, 6, 7, 8],
        "Column3": [9, 10, 11, 12],
    }

    df = pd.DataFrame(data)

    mock_response = MagicMock()
    mock_response.status_code = 200

    tl.utils.upload_dataframe_to_presigned_url(
        df, "https://example.com/api/data", verbose=True, check=False
    )

    captured = capsys.readouterr()

    assert captured.out.startswith("Dataframe is uploading.")
