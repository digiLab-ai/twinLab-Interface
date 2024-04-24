# Standard imports
import io

# Third-party imports
import pandas as pd
import requests
from typeguard import typechecked

# Project imports
from . import settings
from ._version import __version__


@typechecked
def get_message(response: dict) -> str:
    # TODO: This could be a method of the response object
    # TODO: This should be better
    try:
        message = response["message"]
    except:
        message = response
    return message


@typechecked
def get_value_from_body(key: str, body: dict):
    """

    Relay responses from api.py directly.

    This improves error messaging.

    """
    if key in body.keys():
        return body[f"{key}"]
    else:
        print(body)
        raise KeyError(f"{key} not in API response body")


@typechecked
def coerce_params_dict(params: dict) -> dict:
    """

    Relabel parameters to be consistent with twinLab library.

    """
    if "train_test_split" in params.keys() or "test_train_split" in params.keys():
        raise TypeError("train_test_split is deprecated. Use train_test_ratio instead.")
    for param in settings.PARAMS_COERCION:
        if param in params:
            params[settings.PARAMS_COERCION[param]] = params.pop(param)
    if "train_test_ratio" not in params.keys():
        params["train_test_ratio"] = 1.0
    return params


@typechecked
def check_dataset(string: str) -> None:
    """

    Check that a sensible dataframe can be created from a .csv file string.

    """

    # check for duplicate columns # TODO this assumes label is row 0
    header = pd.read_csv(io.StringIO(string), header=None, nrows=1).iloc[0].to_list()
    if len(set(header)) != len(header):
        raise TypeError("Dataset must contain no duplicate column names.")

    string_io = io.StringIO(string)
    try:
        df = pd.read_csv(string_io)
    except Exception:
        raise TypeError("Could not parse the input into a dataframe.")

    # Check that dataset has at least one column.
    if df.shape[0] < 1:
        raise TypeError("Dataset must have at least one column.")

    # Check that dataset has no duplicate column names.
    # TODO: Is this needed? What if the columns with identical names are not used in training?
    if len(set(df.columns)) != len(df.columns):
        raise TypeError(
            "Dataset must contain no duplicate column names."
        )  # Unable to raise this error as column names when read from a string are not recognised?

    # Check that the dataset contains only numerical values.
    if not df.applymap(lambda x: isinstance(x, (int, float))).all().all():
        raise Warning("Dataset contains non-numerical values.")


@typechecked
def upload_dataframe_to_presigned_url(
    df: pd.DataFrame,
    url: str,
    verbose: bool = False,
    check: bool = False,
) -> None:
    """

    Upload a `pandas.DataFrame` to the specified pre-signed URL.

    Args:
        df (pandas.DataFrame): The `pandas.DataFrame` to upload
        url (str): The pre-signed URL generated for uploading the file.
        verbose (bool): defaults to `False`.
        check (bool): defaults to `False`. Check the dataset before uploading.

    """
    if check:
        csv_string = df.to_csv(index=False)
        check_dataset(csv_string)
    headers = {"Content-Type": "application/octet-stream"}
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    buffer = buffer.getvalue()
    response = requests.put(url, data=buffer, headers=headers)
    if verbose:
        if response.status_code == 200:
            print(f"Dataframe is uploading.")
        else:
            print(f"Dataframe upload failed")
            print(f"Status code: {response.status_code}")
            print(f"Reason: {response.text}")


@typechecked
def upload_file_to_presigned_url(
    file_path: str,
    url: str,
    verbose: bool = False,
    check: bool = False,
) -> None:
    """

    Upload a file to the specified pre-signed URL.

    Args:
        file_path (str): The path to the local file a user wants to upload.
        presigned_url (str): The pre-signed URL generated for uploading the file.
        verbose (bool): defaults to `False`.
        check (bool): defaults to `False`. Check the dataset before uploading.

    """
    if check:
        with open(file_path, "rb") as file:
            csv_string = file.read().decode("utf-8")
            check_dataset(csv_string)
    with open(file_path, "rb") as file:
        headers = {"Content-Type": "application/octet-stream"}
        response = requests.put(url, data=file, headers=headers)
    if verbose:
        if response.status_code == 200:
            print(f"File {file_path} is uploading.")
        else:
            print(f"File upload failed")
            print(f"Status code: {response.status_code}")
            print(f"Reason: {response.text}")


@typechecked
def download_dataframe_from_presigned_url(url: str) -> io.StringIO:
    """
    Download a `pandas.DataFrame` from the specified pre-signed URL.

    Args:
        url (str): The pre-signed URL generated for downloading the file.

    Returns:
        io.StringIO: The dataframe in string format downloaded from the URL.

    """
    response = requests.get(url)
    buffer = io.StringIO(response.text)
    return buffer


@typechecked
def get_csv_string(df: pd.DataFrame) -> str:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()


@typechecked
def remove_none_values(d: dict) -> dict:
    """
    Function to remove None values from a nested dictionary.
    """
    return {
        k: remove_none_values(v) if isinstance(v, dict) else v
        for k, v in d.items()
        if v is not None
    }
