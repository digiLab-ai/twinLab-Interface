# Standard imports
import io
import json
import os
import time
from pprint import pprint
from typing import Optional, Tuple, Union

import pandas as pd

# Third-party imports
from deprecated import deprecated
from typeguard import typechecked

# Project imports
from . import api, settings, utils

DEPRECATION_VERSION = "2.5.0"
DEPRECATION_MESSAGE = f"The twinLab Python interface version v1 will be deprecated imminently. Please upgrade to the latest version of the twinLab Python interface. Available at https://pypi.org/project/twinlab/."
PING_TIME_TRAIN_DEFAULT = 5.0  # Seconds
PING_TIME_PREDICT_DEFAULT = 5.0  # Seconds

### Utility functions ###

# TODO: Move to utils.py?


def _get_value_from_body(key: str, body: dict):
    # TODO: Improve the error messaging and relate response from api.py directly here
    # NOTE: Relay responses from api.py directly through this function
    # NOTE: This improves the error messaging
    if key in body.keys():
        return body[f"{key}"]
    else:
        print(body)
        raise KeyError(f"{key} not in API response body")


def _status_campaign(
    campaign_id: str, verbose: Optional[bool] = False, debug: Optional[bool] = False
) -> dict:
    _, response = api.get_status_model(campaign_id, verbose=debug)
    if verbose:
        message = _get_message(response)
        print(message)
    return response


def _get_csv_string(filepath_or_df: Union[str, pd.DataFrame]) -> str:
    if type(filepath_or_df) is str:
        filepath = filepath_or_df
        csv_string = open(filepath, "r").read()
    elif type(filepath_or_df) is pd.DataFrame:
        df = filepath_or_df
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        csv_string = buffer.getvalue()
    else:
        raise ValueError("filepath_or_df must be a string or `pandas.DataFrame`")
    return csv_string


@typechecked
def _use_campaign(
    campaign_id: str,
    method: str,
    filepath_or_df: Optional[Union[str, pd.DataFrame]] = None,
    filepath_or_df_std: Optional[Union[str, pd.DataFrame]] = None,
    processor: Optional[str] = "cpu",
    verbose: Optional[bool] = False,
    debug: Optional[bool] = False,
    **kwargs,
) -> Union[io.StringIO, list, float]:
    if filepath_or_df is not None:
        data_csv = _get_csv_string(filepath_or_df)
    else:
        data_csv = None
    if filepath_or_df_std is not None:
        data_std_csv = _get_csv_string(filepath_or_df_std)
    else:
        data_std_csv = None
    _, response = api.use_model(
        campaign_id,
        method,
        data_csv=data_csv,
        data_std_csv=data_std_csv,
        processor=processor,
        verbose=debug,
        **kwargs,
    )
    if "dataframe" in response.keys():
        output_csv = _get_value_from_body("dataframe", response)
        return io.StringIO(output_csv)
    else:
        output_result = _get_value_from_body("result", response)
        return output_result


def _get_message(response: dict) -> str:
    # TODO: This could be a method of the response object
    # TODO: This should be better
    try:
        message = response["message"]
    except:
        message = response
    return message


### ###

### General functions ###


@deprecated(
    version=DEPRECATION_VERSION,
    reason=DEPRECATION_MESSAGE,
)
@typechecked
def set_api_key(api_key: str) -> None:
    """

    Set the user API key for the `twinLab` cloud.

    Args:
        api_key (str): API key for the `twinLab` cloud.

    Examples:
        .. code-block:: python

            tl.set_api_key("12345")

    .. deprecated:: 2.5.0
            The twinLab Python interface version v1 will be deprecated imminently. Please upgrade to the latest version of the twinLab Python interface. Available at https://pypi.org/project/twinlab/.

    """
    os.environ["TWINLAB_API_KEY"] = api_key


@deprecated(
    version=DEPRECATION_VERSION,
    reason=DEPRECATION_MESSAGE,
)
@typechecked
def set_server_url(url: str) -> None:
    """

    Set server URL

    Set the server URL for the `twinLab` cloud.

    Args:
        url (str): URL for the `twinLab` cloud.

    Examples:
        .. code-block:: python


            tl.set_url("https://twinlab.digilab.co.uk")

    .. deprecated:: 2.5.0
        The twinLab Python interface version v1 will be deprecated imminently. Please upgrade to the latest version of the twinLab Python interface. Available at https://pypi.org/project/twinlab/.
    """
    os.environ["TWINLAB_URL"] = url


@deprecated(
    version=DEPRECATION_VERSION,
    reason=DEPRECATION_MESSAGE,
)
@typechecked
def get_server_url() -> str:
    """

    Get the server URL

    Get the server URL for the `twinLab` cloud.

    Returns:
        str: containing the server URL

    Examples:
        .. code-block:: python

            tl.get_server_url()

    .. deprecated:: 2.5.0
        The twinLab Python interface version v1 will be deprecated imminently. Please upgrade to the latest version of the twinLab Python interface. Available at https://pypi.org/project/twinlab/.

    """
    return os.getenv("TWINLAB_URL")


@deprecated(
    version=DEPRECATION_VERSION,
    reason=DEPRECATION_MESSAGE,
)
@typechecked
def get_api_key() -> str:
    """

    Get the user API key for the `twinLab` cloud.

    Returns:
        str: containing the server API key

    Examples:
        .. code-block:: python

            tl.get_api_key()

    .. deprecated:: 2.5.0
        The twinLab Python interface version v1 will be deprecated imminently. Please upgrade to the latest version of the twinLab Python interface. Available at https://pypi.org/project/twinlab/.

    """
    return os.getenv("TWINLAB_API_KEY")


@deprecated(
    version=DEPRECATION_VERSION,
    reason=DEPRECATION_MESSAGE,
)
@typechecked
def get_user_information(
    verbose: Optional[bool] = False, debug: Optional[bool] = False
) -> dict:
    """

    Get information about the user.

    Args:
        verbose (bool, optional): `Optional`. Determining level of information returned to the user.

    Returns:
        dict: containing user information

    Examples:
        .. code-block:: python

            user_info = tl.get_user_information()
            print(user_info)

    .. deprecated:: 2.5.0
        The twinLab Python interface version v1 will be deprecated imminently. Please upgrade to the latest version of the twinLab Python interface. Available at https://pypi.org/project/twinlab/.

    """
    _, response = api.get_user(verbose=debug)
    user_info = response
    if verbose:
        print("User information:")
        pprint(user_info, compact=True, sort_dicts=False)
    return user_info


@deprecated(
    version=DEPRECATION_VERSION,
    reason=DEPRECATION_MESSAGE,
)
@typechecked
def get_versions(
    verbose: Optional[bool] = False, debug: Optional[bool] = False
) -> dict:
    """


    Get information about the twinLab version being used

    Args:
        verbose (bool, optional): `Optional`. Determining level of information returned to the user.

    Returns:
        dict: Dictionary containing version information

    Examples:
        .. code-block:: python

            version_info = tl.get_versions()
            print(version_info)

    .. deprecated:: 2.5.0
        The twinLab Python interface version v1 will be deprecated imminently. Please upgrade to the latest version of the twinLab Python interface. Available at https://pypi.org/project/twinlab/.

    """
    _, response = api.get_versions(verbose=debug)
    version_info = response
    if verbose:
        print("Version information:")
        pprint(version_info, compact=True, sort_dicts=False)
    return version_info


### ###

### Dataset functions ###


@deprecated(
    version=DEPRECATION_VERSION,
    reason=DEPRECATION_MESSAGE,
)
@typechecked
def upload_dataset(
    filepath_or_df: Union[str, pd.DataFrame],
    dataset_id: str,
    use_upload_url: Optional[bool] = True,
    verbose: Optional[bool] = False,
    debug: Optional[bool] = False,
) -> None:
    """


    Upload a dataset to the `twinLab` cloud so that it can be queried and used for training.

    Args:
        filepath_or_df (Union[`str`.`pandas.DataFrame`]): Location of csv dataset on local machine or `pandas.DataFrame`.
        dataset_id (str): Name for the dataset when saved to the twinLab cloud.
        use_upload_url (bool, optional): `Optional`. Determining whether to upload via a pre-signed url or directly to the server.
        verbose (bool, optional): `Optional`. Determining level of information returned to the user.

    Examples:
        Upload a local file:

        .. code-block:: python

            data_filepath = "path/to/dataset.csv"
            tl.upload_dataset(data_filepath, "my_dataset")

        Upload a `pandas.DataFrame`:

        .. code-block:: python

            df = pd.DataFrame({'X': [1, 2, 3, 4], 'y': [1, 4, 9, 16]})
            tl.upload_dataset(df, "my_dataset")

    .. deprecated:: 2.5.0
        The twinLab Python interface version v1 will be deprecated imminently. Please upgrade to the latest version of the twinLab Python interface. Available at https://pypi.org/project/twinlab/.

    """

    # Upload the file (either via link or directly)
    if use_upload_url:
        _, response = api.generate_upload_url(dataset_id, verbose=debug)
        upload_url = _get_value_from_body("url", response)
        if type(filepath_or_df) is str:
            filepath = filepath_or_df
            utils.upload_file_to_presigned_url(
                filepath, upload_url, verbose=verbose, check=settings.CHECK_DATASETS
            )
        elif type(filepath_or_df) is pd.DataFrame:
            df = filepath_or_df
            utils.upload_dataframe_to_presigned_url(
                df, upload_url, verbose=verbose, check=settings.CHECK_DATASETS
            )
        else:
            raise ValueError("filepath_or_df must be a string or `pandas.DataFrame`")
        if verbose:
            print("Processing dataset.")
        _, response = api.process_uploaded_dataset(dataset_id, verbose=debug)

    else:
        csv_string = _get_csv_string(filepath_or_df)
        _, response = api.upload_dataset(dataset_id, csv_string, verbose=debug)

    if verbose:
        message = _get_message(response)
        print(message)


@deprecated(
    version=DEPRECATION_VERSION,
    reason=DEPRECATION_MESSAGE,
)
@typechecked
def list_datasets(
    verbose: Optional[bool] = False, debug: Optional[bool] = False
) -> list:
    """

    List datasets that have been uploaded to the `twinLab` cloud.

    Args:
        verbose (bool, optional): `Optional`. Determining level of information returned to the user.

    Returns:
        list: List of datasets uploaded to the `twinLab` cloud.

    Examples:
        .. code-block:: python

            datasets = tl.list_datasets()
            print(datasets)

    .. deprecated:: 2.5.0
        The twinLab Python interface version v1 will be deprecated imminently. Please upgrade to the latest version of the twinLab Python interface. Available at https://pypi.org/project/twinlab/.

    """
    _, response = api.list_datasets(verbose=debug)
    datasets = _get_value_from_body("datasets", response)
    if verbose:
        print("Datasets:")
        pprint(datasets, compact=True, sort_dicts=False)
    return datasets


@deprecated(
    version=DEPRECATION_VERSION,
    reason=DEPRECATION_MESSAGE,
)
@typechecked
def view_dataset(
    dataset_id: str, verbose: Optional[bool] = False, debug: Optional[bool] = False
) -> pd.DataFrame:
    """

    View dataset

    View a dataset that exists on the twinLab cloud.

    Args:
        dataset_id (str): Name for the dataset when saved to the twinLab cloud.
        verbose (bool, optional): `Optional`. Determining level of information returned to the user.

    Returns:
        pandas.DataFrame: Dataframe of the dataset

    Examples:
        .. code-block:: python

            df = pd.DataFrame({'X': [1, 2, 3, 4], 'y': [1, 4, 9, 16]})
            tl.upload_dataset(df, "my_dataset")
            df = tl.view_dataset("my_dataset")
            print(df)

    .. deprecated:: 2.5.0
        The twinLab Python interface version v1 will be deprecated imminently. Please upgrade to the latest version of the twinLab Python interface. Available at https://pypi.org/project/twinlab/.

    """
    _, response = api.view_dataset(dataset_id, verbose=debug)
    csv_string = _get_value_from_body("dataset", response)
    csv_string = io.StringIO(csv_string)
    df = pd.read_csv(csv_string, sep=",")
    if verbose:
        print("Dataset:")
        print(df)
    return df


@deprecated(
    version=DEPRECATION_VERSION,
    reason=DEPRECATION_MESSAGE,
)
@typechecked
def query_dataset(
    dataset_id: str, verbose: Optional[bool] = False, debug: Optional[bool] = False
) -> pd.DataFrame:
    """

    Query dataset

    Query a dataset that exists on the `twinLab` cloud by printing summary statistics.

    Args:
        dataset_id (str): Name of dataset on S3. This should be the same as the uploaded file name.
        verbose (bool, optional): `Optional`. Determining level of information returned to the user.

    Returns:
        pandas.DataFrame: Dataframe containing summary statistics for the dataset

    Examples:
        .. code-block:: python

            df = pd.DataFrame({'X': [1, 2, 3, 4], 'y': [1, 4, 9, 16]})
            tl.upload_dataset(df, "my_dataset")
            df = tl.query_dataset("my_dataset")
            print(df)

    .. deprecated:: 2.5.0
        The twinLab Python interface version v1 will be deprecated imminently. Please upgrade to the latest version of the twinLab Python interface. Available at https://pypi.org/project/twinlab/.

    """
    _, response = api.summarise_dataset(dataset_id, verbose=debug)
    csv_string = _get_value_from_body("dataset_summary", response)
    csv_string = io.StringIO(csv_string)
    df = pd.read_csv(csv_string, index_col=0, sep=",")
    if verbose:
        print("Dataset summary:")
        print(df)
    return df


@deprecated(
    version=DEPRECATION_VERSION,
    reason=DEPRECATION_MESSAGE,
)
@typechecked
def delete_dataset(
    dataset_id: str, verbose: Optional[bool] = False, debug: Optional[bool] = False
) -> None:
    """

    Delete dataset

    Delete a dataset from the `twinLab` cloud.

    Args:
        dataset_id (str): Name of dataset to delete from the Cloud.
        verbose (bool, optional): `Optional`. Determining level of information returned to the user.

    Examples:
        .. code-block:: python

            df = pd.DataFrame({'X': [1, 2, 3, 4], 'y': [1, 4, 9, 16]})
            tl.upload_dataset(df, "my_dataset")
            tl.delete_dataset("my_dataset")

    .. deprecated:: 2.5.0
        The twinLab Python interface version v1 will be deprecated imminently. Please upgrade to the latest version of the twinLab Python interface. Available at https://pypi.org/project/twinlab/.

    """
    _, response = api.delete_dataset(dataset_id, verbose=debug)
    if verbose:
        message = _get_message(response)
        print(message)


###  ###

### Campaign functions ###


@deprecated(
    version=DEPRECATION_VERSION,
    reason=DEPRECATION_MESSAGE,
)
@typechecked
def train_campaign(
    filepath_or_params: Union[str, dict],
    campaign_id: str,
    ping_time: Optional[float] = PING_TIME_TRAIN_DEFAULT,
    processor: Optional[str] = "cpu",
    verbose: Optional[bool] = False,
    debug: Optional[bool] = False,
) -> None:
    """

    Train a campaign in the `twinLab` cloud.

    Args:
        filepath_or_params (str): `dict`. Union. Filepath to local json or parameters dictionary for training.
        campaign_id (str): Name for the final trained campaign.
        verbose (bool, optional): `Optional`. Determining level of information returned to the user.
        dataset_id (str): Dataset ID of the dataset as stored in the Cloud.
        inputs (list): A list of strings referring to the columns in the
        outputs (list): A list of strings referring to the columns in the
        estimator (str, optional): `Optional`. The type of estimator used in the pipeline. This can be either
        estimator_kwargs (dict, optional): `Optional`. Keywords passed to the underlying estimator.
        decompose_input (bool, optional): `Optional`. Specifies whether the input parameters
        input_explained_variance (float, optional): `Optional`. Specifies how much of the
        decompose_output (bool, optional): `Optional`. Specifies whether the output parameters
        output_explained_variance (float, optional): `Optional`. Specifies how much of the
        train_test_ratio (float, optional): `Optional`. Specifies the ratio of training samples in
        model_selection (bool, optional): `Optional`. Whether to run model selection.
        model_selection_kwargs (dict, optional): `Optional`. Keywords passed to the model
        seed (int, optional): `Optional`. Specifies the seed for the random number generator.

    Examples:

        Train using a local `.json` parameters file:

        .. code-block:: python

            import twinlab as tl

            tl.train_campaign("path/to/params.json", "my_campaign")

        Train via a `python` dictionary:

        .. code-block:: python

            import pandas as pd
            import twinlab as tl

            df = pd.DataFrame({'X': [1, 2, 3, 4], 'y': [1, 4, 9, 16]})
            tl.upload_dataset(df, "my_dataset")
            params = {
                "dataset_id": "my_dataset",
                "inputs": ["X"],
                "outputs": ["y"],
            }
            tl.train_campaign(params, "my_campaign")


    .. deprecated:: 2.5.0
        The twinLab Python interface version v1 will be deprecated imminently. Please upgrade to the latest version of the twinLab Python interface. Available at https://pypi.org/project/twinlab/.

    """
    if type(filepath_or_params) is dict:
        params = filepath_or_params
    elif type(filepath_or_params) is str:
        filepath = filepath_or_params
        params = json.load(open(filepath))
    else:
        print("Type:", type(filepath_or_params))
        raise ValueError("filepath_or_params must be either a string or a dictionary")
    params = utils.coerce_params_dict(params)
    params_str = json.dumps(params)
    _, response = api.train_model(
        campaign_id, params_str, processor=processor, verbose=debug
    )
    if verbose:
        message = _get_message(response)
        print(message)

    # Wait for job to complete
    complete = False
    while not complete:
        status = _status_campaign(campaign_id, verbose=False, debug=debug)
        complete = _get_value_from_body("job_complete", status)
        time.sleep(ping_time)


@deprecated(
    version=DEPRECATION_VERSION,
    reason=DEPRECATION_MESSAGE,
)
@typechecked
def list_campaigns(
    verbose: Optional[bool] = False, debug: Optional[bool] = False
) -> list:
    """

    List campaigns

    List campaigns that have been completed to the `twinLab` cloud.

    Args:
        verbose (bool, optional): `Optional`. Determining level of information returned to the user.

    Returns:
        list

    Examples:
        .. code-block:: python

            import twinlab as tl

            campaigns = tl.list_campaigns()
            print(campaigns)


    .. deprecated:: 2.5.0
        The twinLab Python interface version v1 will be deprecated imminently. Please upgrade to the latest version of the twinLab Python interface. Available at https://pypi.org/project/twinlab/.

    """
    _, response = api.list_models(verbose=debug)
    campaigns = _get_value_from_body("models", response)
    if verbose:
        print("Trained models:")
        pprint(campaigns, compact=True, sort_dicts=False)
    return campaigns


@deprecated(
    version=DEPRECATION_VERSION,
    reason=DEPRECATION_MESSAGE,
)
@typechecked
def view_campaign(
    campaign_id: str, verbose: Optional[bool] = False, debug: Optional[bool] = False
) -> dict:
    """

    View campaign

    View a campaign that exists on the twinLab cloud.

    Args:
        campaign_id (str): Name for the model when saved to the twinLab cloud.
        verbose (bool, optional): `Optional`. Determining level of information returned to the user.

    Returns:
        dict: containing the campaign training parameters

    Examples:
        .. code-block:: python

            import pandas as pd
            import twinlab as tl

            df = pd.DataFrame({'X': [1, 2, 3, 4], 'y': [1, 4, 9, 16]})
            tl.upload_dataset(df, "my_dataset")
            params = {
                "dataset_id": "my_dataset",
                "inputs": ["X"],
                "outputs": ["y"],
            }
            tl.train_campaign(params, "my_campaign")
            params = tl.view_campaign("my_campaign")
            print(params)

    .. deprecated:: 2.5.0
        The twinLab Python interface version v1 will be deprecated imminently. Please upgrade to the latest version of the twinLab Python interface. Available at https://pypi.org/project/twinlab/.

    """
    _, response = api.view_model(campaign_id, verbose=debug)
    model_parameters = response
    if verbose:
        print("Campaign summary:")
        pprint(model_parameters, compact=True, sort_dicts=False)
    return model_parameters


@deprecated(
    version=DEPRECATION_VERSION,
    reason=DEPRECATION_MESSAGE,
)
@typechecked
def query_campaign(
    campaign_id: str, verbose: Optional[bool] = False, debug: Optional[bool] = False
) -> dict:
    """

    Query campaign

    Get summary statistics for a pre-trained campaign in the `twinLab` cloud.

    Args:
        campaign_id (str): Name of trained campaign to query
        verbose (bool, optional): `Optional`. Determining level of information returned to the user.

    Returns:
        dict: containing summary statistics for the pre-trained campaign

    Examples:
        .. code-block:: python

            df = pd.DataFrame({'X': [1, 2, 3, 4], 'y': [1, 4, 9, 16]})
            tl.upload_dataset(df, "my_dataset")
            params = {
                "dataset_id": "my_dataset",
                "inputs": ["X"],
                "outputs": ["y"],
            }
            tl.train_campaign(params, "my_campaign")
            info = tl.query_campaign("my_campaign")
            print(info)

    .. deprecated:: 2.5.0
        The twinLab Python interface version v1 will be deprecated imminently. Please upgrade to the latest version of the twinLab Python interface. Available at https://pypi.org/project/twinlab/.

    """
    _, response = api.summarise_model(campaign_id, verbose=debug)
    summary = response
    if verbose:
        print("Model summary:")
        pprint(summary, compact=True, sort_dicts=False)
    return summary


@deprecated(
    version=DEPRECATION_VERSION,
    reason=DEPRECATION_MESSAGE,
)
@typechecked
def predict_campaign(
    filepath_or_df: Union[str, pd.DataFrame],
    campaign_id: str,
    sync: Optional[bool] = False,
    ping_time: Optional[float] = PING_TIME_PREDICT_DEFAULT,
    processor: Optional[str] = "cpu",
    verbose: Optional[bool] = False,
    debug: Optional[bool] = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """

    Predict campaign

    Make predictions from a pre-trained model that exists on the `twinLab` cloud.

    Args:
        filepath_or_df (str): Location of .csv file dataset on local machine
        campaign_id (str): Name of pre-trained campaign to use for predictions.
        verbose (bool, optional): `Optional`. Determining level of information returned to the user.
        campaign_id (str`.):

    Returns:
        tuple: containing
        df_mean:
        df_std:

    Examples:

        Using a local file:

        .. code-block:: python

            import pandas as pd
            import twinlab as tl

            df = pd.DataFrame({'X': [1, 2, 3, 4], 'y': [1, 4, 9, 16]})
            tl.upload_dataset(df, "my_dataset")
            params = {
                "dataset_id": "my_dataset",
                "inputs": ["X"],
                "outputs": ["y"],
            }
            tl.train_campaign(params, "my_campaign")
            filepath = "path/to/data.csv" # Local
            campaign_id = 'my_campaign" # Pre-trained
            df_mean, df_std = tl.predict_campaign(filepath, campaign_id)
            print(df_mean)
            print(df_std)

        Using a `pandas.DataFrame`:

        .. code-block:: python

            import pandas as pd
            import twinlab as tl

            df = pd.DataFrame({'X': [1, 2, 3, 4], 'y': [1, 4, 9, 16]})
            tl.upload_dataset(df, "my_dataset")
            params = {
                "dataset_id": "my_dataset",
                "inputs": ["X"],
                "outputs": ["y"],
            }
            tl.train_campaign(params, "my_campaign")
            df = pd.DataFrame({'X': [1.5, 2.5, 3.5]})
            tl.predict_campaign(df, "my_campaign")

    .. deprecated:: 2.5.0
        The twinLab Python interface version v1 will be deprecated imminently. Please upgrade to the latest version of the twinLab Python interface. Available at https://pypi.org/project/twinlab/.

    """

    if sync:
        csv = _use_campaign(
            campaign_id,
            method="predict",
            filepath_or_df=filepath_or_df,
            processor=processor,
            verbose=verbose,
            debug=debug,
        )
    else:
        # Send off the request to predict
        data_csv = _get_csv_string(filepath_or_df)
        _, response = api.use_request_model(
            campaign_id, "predict", data_csv, processor=processor, verbose=debug
        )
        process_id = _get_value_from_body("process_id", response)

        # Wait for job to complete
        status = 202
        while status == 202:
            status, response = api.use_response_model(
                campaign_id, "predict", process_id, verbose=debug
            )
            time.sleep(ping_time)
        csv = _get_value_from_body("dataframe", response)
        csv = io.StringIO(csv)

    # Munge the response into the mean and std
    df = pd.read_csv(csv, sep=",")
    n = len(df.columns)
    df_mean, df_std = df.iloc[:, : n // 2], df.iloc[:, n // 2 :]
    df_std.columns = df_std.columns.str.removesuffix(" [std_dev]")
    if verbose:
        print("Mean predictions:")
        print(df_mean)
        print("Standard deviation predictions:")
        print(df_std)

    return df_mean, df_std


@deprecated(
    version=DEPRECATION_VERSION,
    reason=DEPRECATION_MESSAGE,
)
@typechecked
def sample_campaign(
    filepath_or_df: Union[str, pd.DataFrame],
    campaign_id: str,
    num_samples: int,
    processor: Optional[str] = "cpu",
    verbose: Optional[bool] = False,
    debug: Optional[bool] = False,
    **kwargs,
) -> pd.DataFrame:
    """

    Sample campaign

    Draw samples from a pre-trained campaign that exists on the `twinLab` cloud.

    Args:
        filepath_or_df (str): Location of .csv dataset on local machine for evaluation
        campaign_id (str): Name of pre-trained campaign to use for predictions.
        num_samples (int): Number of samples to draw for each row of the evaluation data.
        verbose (bool, optional): `Optional`. Determining level of information returned to the user.

    Returns:
        pandas.DataFrame: with the sampled values

    Examples:
        Using a local file:

        .. code-block:: python

            import pandas as pd
            import twinlab as tl

            df = pd.DataFrame({'X': [1, 2, 3, 4], 'y': [1, 4, 9, 16]})
            tl.upload_dataset(df, "my_dataset")
            params = {
                "dataset_id": "my_dataset",
                "inputs": ["X"],
                "outputs": ["y"],
            }
            tl.train_campaign(params, "my_campaign")
            filepath = "path/to/data.csv" # Local
            n = 10
            df_mean, df_std = tl.sample_campaign(filepath, "my_campaign", n)
            print(df_mean)
            print(df_std)

        Using a `pandas.DataFrame`:

        .. code-block:: python

            import pandas as pd
            import twinlab as tl

            df = pd.DataFrame({'X': [1, 2, 3, 4], 'y': [1, 4, 9, 16]})
            tl.upload_dataset(df, "my_dataset")
            params = {
                "dataset_id": "my_dataset",
                "inputs": ["X"],
                "outputs": ["y"],
            }
            tl.train_campaign(params, "my_campaign")
            df = pd.DataFrame({'X': [1.5, 2.5, 3.5]})
            n = 10
            tl.sample_campaign(df, "my_campaign", n)

    .. deprecated:: 2.5.0
        The twinLab Python interface version v1 will be deprecated imminently. Please upgrade to the latest version of the twinLab Python interface. Available at https://pypi.org/project/twinlab/.

    """

    csv = _use_campaign(
        campaign_id,
        method="sample",
        filepath_or_df=filepath_or_df,
        num_samples=num_samples,
        processor=processor,
        verbose=verbose,
        debug=debug,
        **kwargs,
    )
    df = pd.read_csv(csv, header=[0, 1], sep=",")
    if verbose:
        print("Samples:")
        print(df)
    return df


@deprecated(
    version=DEPRECATION_VERSION,
    reason=DEPRECATION_MESSAGE,
)
@typechecked
def active_learn_campaign(
    campaign_id: str,
    num_points: int,
    processor: Optional[str] = "cpu",
    verbose: Optional[bool] = False,
    debug: Optional[bool] = False,
    **kwargs,
) -> pd.DataFrame:
    """

    Active learn campaign

    Draw new candidate data points via active learning from a pre-trained campaign

    that exists on the `twinLab` cloud.

    Args:
        campaign_id (str): Name of pre-trained campaign to use for predictions.
        num_points (int): Number of samples to draw for each row of the evaluation data.
        verbose (bool, optional): `Optional`. Determining level of information returned to the user.

    Returns:
        pandas.DataFrame: containing the recommended sample locations

    Examples:
        .. code-block:: python

            import pandas as pd
            import twinlab as tl

            df = pd.DataFrame({'X': [1, 2, 3, 4], 'y': [1, 4, 9, 16]})
            tl.upload_dataset(df, "my_dataset")
            params = {
                "dataset_id": "my_dataset",
                "inputs": ["X"],
                "outputs": ["y"],
            }
            tl.train_campaign(params, "my_campaign")
            n = 10
            df = tl.active_learn_campaign("my_campaign", n)
            print(df)

    .. deprecated:: 2.5.0
        The twinLab Python interface version v1 will be deprecated imminently. Please upgrade to the latest version of the twinLab Python interface. Available at https://pypi.org/project/twinlab/.

    """

    csv = _use_campaign(
        campaign_id,
        method="get_candidate_points",
        acq_func="qNIPV",
        num_points=num_points,
        processor=processor,
        verbose=verbose,
        debug=debug,
        **kwargs,
    )
    df = pd.read_csv(csv, sep=",")
    if verbose:
        print("Candidate points:")
        print(df)
    return df


@deprecated(
    version=DEPRECATION_VERSION,
    reason=DEPRECATION_MESSAGE,
)
@typechecked
def optimise_campaign(
    campaign_id: str,
    num_points: int,
    processor: Optional[str] = "cpu",
    verbose: Optional[bool] = False,
    debug: Optional[bool] = False,
    **kwargs,
) -> pd.DataFrame:
    """

    Optimise campaign

    Draw new candidate data points by optimizing for "qEI" (Monte Carlo Expected Improvement)

    acquisition function from a pre-trained campaign that exists on the `twinLab` cloud.

    Args:
        campaign_id (str): Name of pre-trained campaign to use for predictions.
        num_points (int): Number of samples to draw for each row of the evaluation data.
        verbose (bool, optional): `Optional`. Determining level of information returned to the user.
        acq_kwargs (dict, optional): `Optional`. Specifies the keyword arguments to modify the behavior of
        weights (list[float]): Specifies the weightage for different objectives to be

    Returns:
        pandas.DataFrame: containing the recommended sample locations

    Examples:
        .. code-block:: python

            import pandas as pd
            import twinlab as tl

            df = pd.DataFrame({'X': [0.0, 0.25, 0.75, 1.0], 'y': [-1.60856306, -0.27526546, -0.34670215, -1.65062947]})
            tl.upload_dataset(df, "my_dataset")
            params = {
                "dataset_id": "my_dataset",
                "inputs": ["X"],
                "outputs": ["y"],
            }
            tl.train_campaign(params, "my_campaign")
            n = 1
            df = tl.optimise_campaign("my_campaign", n)
            print(df)

    .. deprecated:: 2.5.0
        The twinLab Python interface version v1 will be deprecated imminently. Please upgrade to the latest version of the twinLab Python interface. Available at https://pypi.org/project/twinlab/.

    """

    csv = _use_campaign(
        campaign_id,
        method="get_candidate_points",
        acq_func="qEI",
        num_points=num_points,
        processor=processor,
        verbose=verbose,
        debug=debug,
        **kwargs,
    )
    df = pd.read_csv(csv, sep=",")
    if verbose:
        print("Candidate points:")
        print(df)
    return df


@deprecated(
    version=DEPRECATION_VERSION,
    reason=DEPRECATION_MESSAGE,
)
@typechecked
def solve_inverse_campaign(
    campaign_id: str,
    filepath_or_df: Union[str, pd.DataFrame],
    filepath_or_df_std: Union[str, pd.DataFrame],
    processor: Optional[str] = "cpu",
    verbose: Optional[bool] = False,
    debug: Optional[bool] = False,
    **kwargs,
) -> pd.DataFrame:
    """

    Inverse modelling on campaign

    Given a set of observations, inverse modelling finds the model that would best suit the data.

    Args:
        campaign_id (str): Name of pre-trained campaign to use for predictions.
        data_csv (pandas.DataFrame): A `pandas.DataFrame` of observations.
        data_std_csv`  (pandas.DataFrame): A `pandas.DataFrame` of errors on the observations.
        verbose (bool): Determining level of information returned to the user.

    Returns:
        pandas.DataFrame: containing the recommended model statistics

    Examples:
        .. code-block:: python

            import pandas as pd
            import twinlab as tl

            df = pd.DataFrame({'X': [1, 2, 3, 4], 'y': [1, 4, 9, 16]})
            tl.upload_dataset(df, "my_dataset")
            params = {
                "dataset_id": "my_dataset",
                "inputs": ["X"],
                "outputs": ["y"],
            }
            tl.train_campaign(params, "my_campaign")
            data_csv = pd.DataFrame({'y': [1]})
            data_std_csv = pd.DataFrame({'y': [0.498]})
            df = tl.solve_inverse_campaign("my_campaign", data_csv, data_std_csv)
            print(df)

    .. deprecated:: 2.5.0
        The twinLab Python interface version v1 will be deprecated imminently. Please upgrade to the latest version of the twinLab Python interface. Available at https://pypi.org/project/twinlab/.

    """

    csv = _use_campaign(
        campaign_id,
        method="solve_inverse",
        filepath_or_df=filepath_or_df,
        filepath_or_df_std=filepath_or_df_std,
        processor=processor,
        verbose=verbose,
        debug=debug,
        **kwargs,
    )

    # Munge the response into the mean and std
    df = pd.read_csv(csv, sep=",")

    df = df.set_index("Unnamed: 0")
    df.index.name = None
    if "Unnamed: 0.1" in df.columns:
        df = df.drop("Unnamed: 0.1", axis=1)

    if verbose:
        print("Inverse model statistics:")
        print(df)

    return df


@deprecated(
    version=DEPRECATION_VERSION,
    reason=DEPRECATION_MESSAGE,
)
@typechecked
def score_campaign(
    campaign_id: str,
    combined_score: Optional[bool] = False,
    processor: Optional[str] = "cpu",
    verbose: Optional[bool] = True,
    debug: Optional[bool] = False,
) -> Optional[Union[float, pd.DataFrame]]:
    """

    Quantify the performance of your trained model with a model score.

    Args:
        combined_score (bool): Determining whether to average scores across dimensions. If False. will return a `numpy` array even if only one dimension.

    Returns:
        None:
        float:
        pandas.DataFrame: containing the trained model score

    Examples:
        .. code-block:: python

            import pandas as pd
            import twinlab as tl

            df = pd.DataFrame({'X': [1.5, 2.5, 3.5]})
            tl.predict_campaign(df, "my_campaign")
            tl.score_campaign(df, combined_score=True)

    .. deprecated:: 2.5.0
        The twinLab Python interface version v1 will be deprecated imminently. Please upgrade to the latest version of the twinLab Python interface. Available at https://pypi.org/project/twinlab/.

    """

    score = _use_campaign(
        campaign_id,
        method="score",
        combined_score=combined_score,
        processor=processor,
        verbose=verbose,
        debug=debug,
    )
    if not combined_score:
        score = pd.read_csv(score, sep=",")
    if verbose:
        print("Campaign Score:")
        print(score)
    return score


@deprecated(
    version=DEPRECATION_VERSION,
    reason=DEPRECATION_MESSAGE,
)
@typechecked
def get_calibration_curve_campaign(
    campaign_id: str,
    type: Optional[str] = "quantile",
    processor: Optional[str] = "cpu",
    verbose: Optional[bool] = True,
    debug: Optional[bool] = False,
) -> Optional[pd.DataFrame]:
    """

    Quantify the performance of your trained model with a calibration curve.

    Args:
        type (str): Determine whether to use "quantiile" or "interval" for the model calibration error.

    Returns:
        None:
        pd.DataFrame: containing the data for the calibration curve

    Examples:
        .. code-block:: python

            import pandas as pd
            import twinlab as tl

            df = pd.DataFrame({'X': [1.5, 2.5, 3.5]})
            tl.train_campaign(df, "my_campaign")
            tl.get_calibration_curve(df, type="quantile")

            # Plot the calibration curve

            fraction_observed = tl.get_calibration_curve(type="quantile")
            fraction_expected = np.linspace(0,1,fraction_observed.shape[0])

            fig, ax = plt.subplots(figsize=(5, 5))

            ax.set_title("Calibration curve")
            ax.set_xlabel("Expected coverage")
            ax.set_ylabel("Observed coverage")
            plt.plot(np.linspace(0, 1, 100), fraction_observed)
            plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), "--")
            plt.show()

    .. deprecated:: 2.5.0
        The twinLab Python interface version v1 will be deprecated imminently. Please upgrade to the latest version of the twinLab Python interface. Available at https://pypi.org/project/twinlab/.

    """

    csv = _use_campaign(
        campaign_id,
        method="get_calibration_curve",
        type=type,
        processor=processor,
        verbose=verbose,
        debug=debug,
    )
    if csv is not None:
        df = pd.read_csv(csv, sep=",")
        if verbose:
            print("Calibration curve information:")
            pprint(df)
        return df


@deprecated(
    version=DEPRECATION_VERSION,
    reason=DEPRECATION_MESSAGE,
)
@typechecked
def delete_campaign(
    campaign_id: str, verbose: Optional[bool] = False, debug: Optional[bool] = False
) -> None:
    """

    Delete campaign

    Delete campaign from the `twinLab` cloud.

    Args:
        campaign_id (str:):
        verbose (bool, optional): Optional. determining level of information returned to the user.

    Examples:
        .. code-block:: python

            import pandas as pd
            import twinlab as tl

            df = pd.DataFrame({'X': [1, 2, 3, 4], 'y': [1, 4, 9, 16]})
            tl.upload_dataset(df, "my_dataset")
            params = {
                "dataset_id": "my_dataset",
                "inputs": ["X"],
                "outputs": ["y"],
            }
            tl.train_campaign(params, "my_campaign")
            tl.delete_campaign("my_campaign")

    .. deprecated:: 2.5.0
        The twinLab Python interface version v1 will be deprecated imminently. Please upgrade to the latest version of the twinLab Python interface. Available at https://pypi.org/project/twinlab/.

    """
    _, response = api.delete_model(campaign_id, verbose=debug)
    if verbose:
        message = _get_message(response)
        print(message)


### ###
