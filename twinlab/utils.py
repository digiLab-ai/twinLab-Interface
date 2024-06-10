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

    # Check for duplicate columns
    # TODO: This assumes label is row 0
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


@typechecked
def reformat_summary_dict(summary_dict: dict, detailed: bool = False) -> dict:
    """
    Function to reformat the summary dictionary.
    """
    properties, mean, kernel, likelihood = {}, {}, {}, {}

    # To retrieve the dictionary from the "base_estimator_diagnostics" key, when input or output decomposition is applied
    if "base_estimator_diagnostics" in summary_dict["estimator_diagnostics"].keys():
        estimator_diagnostics = summary_dict["estimator_diagnostics"].get(
            "base_estimator_diagnostics"
        )
    else:
        estimator_diagnostics = summary_dict.get("estimator_diagnostics")

    # Non-detailed case when only the learned hyperparameters are returned along with the covar_module and mean_module
    if "learned_hyperparameters" in estimator_diagnostics.keys():
        learned_params = estimator_diagnostics.get("learned_hyperparameters")

        # Augment properties with parameters of the variational distribution for Variational GP
        if "variational_strategy.inducing_points" in learned_params.keys():
            properties["inducing_points"] = learned_params.get(
                "variational_strategy.inducing_points"
            )
            properties["variational_distribution_mean"] = learned_params.get(
                "variational_strategy._variational_distribution.variational_mean"
            )
            properties["variational_distribution_covar"] = learned_params.get(
                "variational_strategy._variational_distribution.chol_variational_covar"
            )

        # Write the standard learned hyperparameters to the properties dictionary
        properties["covariance_noise"] = learned_params.get(
            "likelihood.noise_covar.original_noise"
        )
        mean["mean"] = learned_params.get("mean_module.original_constant")
        mean["mean_function_used"] = estimator_diagnostics.get("mean_module")
        kernel["lengthscale"] = learned_params.get(
            "covar_module.base_kernel.original_lengthscale"
        )
        kernel["outputscale"] = learned_params.get("covar_module.original_outputscale")
        kernel["kernel_function_used"] = estimator_diagnostics["covar_module"].replace(
            "\n", ""
        )
    else:
        return summary_dict
    if not detailed:
        return {"properties": properties, "mean": mean, "kernel": kernel}
    else:
        transform_list = ["input_transform_parameters", "outcome_transform_parameters"]
        join_delimiter = "_"

        # Extracting transform parameters
        if (
            transform_list[0] in estimator_diagnostics.keys()
            and transform_list[1] in estimator_diagnostics.keys()
        ):
            for transform_parameters in transform_list:
                for key in estimator_diagnostics[transform_parameters].keys():
                    # To avoid double underscores for some transform parameters
                    param_name = key.split(".")
                    if param_name[1][0] == "_":
                        param_name = param_name[0] + param_name[1]
                    else:
                        param_name = param_name[0] + "_" + param_name[1]
                    properties[param_name] = estimator_diagnostics[
                        transform_parameters
                    ].get(key)
            properties.pop(
                "outcome_transform_is_trained", None
            )  # Remove outcome_transform_is_trained parameter

        # Extract mean parameters
        if "mean_module_parameters" in estimator_diagnostics.keys():
            mean["raw_constant"] = estimator_diagnostics["mean_module_parameters"].get(
                "mean_module.raw_constant"
            )

        # Extracting kernel parameters
        if "covar_module_parameters" in estimator_diagnostics.keys():
            covar_module_params = estimator_diagnostics["covar_module_parameters"]
            for key in covar_module_params.keys():
                param_name = key.split(".")
                # Retrieving parameter names for Multi-Fidelity GP
                if len(param_name) > 5 and param_name[3] == "0":
                    param_name = join_delimiter.join(param_name[4:])
                # Retrieving parameter names for Mixed Single Task GP
                elif len(param_name) > 6 and param_name[1] == "kernels":
                    param_name = join_delimiter.join(param_name[2:])
                else:
                    del param_name[0]
                    param_name = join_delimiter.join(param_name)
                kernel[param_name] = covar_module_params.get(key)

        # Extracting likelihood parameters
        if "likelihood_parameters" in estimator_diagnostics.keys():
            likelihood_params = estimator_diagnostics["likelihood_parameters"]
            for key in likelihood_params.keys():
                param_name = key.split(".")
                # Shorten names differently for heteroskedastic noise model and other models
                if len(param_name) < 5:
                    param_name = join_delimiter.join(param_name[2:])
                else:
                    param_name = join_delimiter.join(param_name[3:])
                if (
                    param_name[0] == "_"
                ):  # To avoid double underscores for some likelihood parameters
                    likelihood["noise_model" + param_name] = likelihood_params.get(key)
                else:
                    likelihood["noise_model_" + param_name] = likelihood_params.get(key)
        return {
            "properties": properties,
            "mean": mean,
            "kernel": kernel,
            "likelihood": likelihood,
        }
