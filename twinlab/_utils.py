import io
import os
import json
import time
import warnings
from typing import Optional
from datetime import datetime, timedelta
from pprint import pprint

import pandas as pd
import requests
from typeguard import typechecked

from ._version import __version__
from .settings import ValidStatus
from ._api import get_account, get_projects

ALLOWED_DATAFRAME_SIZE = 5.5 * int(1e6)

PING_TIME_INITIAL = 1.0  # Seconds
PING_FRACTIONAL_INCREASE = 0.1

DATETIME_STRING_FORMAT = "%Y-%m-%d %H:%M:%S"


@typechecked
def get_project_id(project_name: str, project_owner_email: str):

    # Make api calls to get the project and project owner
    _, project_owner_account = get_account(project_owner_email)

    _, available_projects = get_projects()
    available_projects = available_projects["projects"]

    # List comprehension to find the matching dictionary
    matching_project = next(
        (
            proj  # Use a different variable name here
            for proj in available_projects
            if proj["name"] == project_name
            and proj["owner_id"] == project_owner_account["_id"]
        ),
        None,  # Default value if no match is found
    )

    if not matching_project:
        raise ValueError("No project found with the given name and owner.")
    else:
        project_id = str(matching_project["_id"])

    return project_id


@typechecked
def match_project(project_name: str, project_owner_email: Optional[str] = None) -> str:
    if project_name == "personal" and project_owner_email is None:
        project_id = "personal"
    else:

        if not project_owner_email:
            project_owner_email = os.getenv("TWINLAB_USER")

        project_id = get_project_id(project_name, project_owner_email)

    return project_id


@typechecked
def _calculate_ping_time(elapsed_time: float) -> float:
    # This smoothly transitions between regular pinging at the initial ping time
    # to more drawn out pinging (expoential) as time goes on
    return PING_TIME_INITIAL + elapsed_time * PING_FRACTIONAL_INCREASE


@typechecked
def wait_for_job_completion(
    api_function: callable, *args, verbose: bool = False
) -> dict:
    start_time = time.time()
    status = ValidStatus.PROCESSING
    while status == ValidStatus.PROCESSING:
        elapsed_time = time.time() - start_time  # This will be ~0 seconds initially
        wait_time = _calculate_ping_time(elapsed_time)
        time.sleep(wait_time)
        _, response = api_function(*args)
        status = ValidStatus(get_value_from_body("status", response))
        if verbose:
            t = timedelta(seconds=elapsed_time)
            t -= timedelta(microseconds=t.microseconds)  # Remove microseconds
            print(f"{str(t)}: Job status: {status.value}")
    return response


@typechecked
def calculate_run_time(start_time_str: str, end_time_str: str) -> str:
    # convert the start_time and end_time to datetime
    start_time = datetime.fromisoformat(start_time_str)
    end_time = datetime.fromisoformat(end_time_str)

    # calculate the run time
    run_time = end_time - start_time

    # round the run time to the nearest second
    run_time_seconds = run_time.total_seconds()
    run_time_rounded = round(run_time_seconds)
    run_time = timedelta(seconds=run_time_rounded)

    # format the run time in a nice format
    run_time_nice_format = str(run_time)

    return run_time_nice_format


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
    # Relat responses from api.py directly
    # This improves error messaging
    if key in body.keys():
        return body[f"{key}"]
    else:
        raise KeyError(f"{key} not in API response body")


@typechecked
def calculate_runtime(start_time: str, end_time: str) -> str:
    # Calculate the runtime of a job given the start and end times.

    start_datetime = datetime.fromisoformat(start_time)
    end_datetime = datetime.fromisoformat(end_time)
    runtime = end_datetime - start_datetime

    # Extract hours, minutes, and seconds from the timedelta object
    total_seconds = int(runtime.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Format the runtime as HH:MM:SS
    nice_time_string = f"{hours:02}:{minutes:02}:{seconds:02}"

    return nice_time_string


@typechecked
def convert_time_format(time_string: str) -> str:
    # Parse the time string into a datetime object
    dt = datetime.fromisoformat(time_string)

    # Format the datetime object into a nice string
    nice_time_string = dt.strftime(DATETIME_STRING_FORMAT)

    return nice_time_string


@typechecked
def convert_time_formats_in_status(status: dict) -> dict:

    start_time = status.get("start_time", None)
    if start_time:
        status["start_time"] = convert_time_format(start_time)
    end_time = status.get("end_time", None)
    if end_time:
        status["end_time"] = convert_time_format(end_time)
    return status


@typechecked
def check_dataset(string: str) -> None:
    # Check that a sensible dataframe can be created from a .csv file string.

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
    if not df.map(lambda x: isinstance(x, (int, float))).all().all():
        raise Warning("Dataset contains non-numerical values.")


@typechecked
def download_file_from_url(presigned_url: str, file_path: str) -> None:
    try:
        # Make the HTTP GET request to fetch the file
        response = requests.get(presigned_url)

        # Write the content to the specified file path
        with open(file_path, "wb") as file:
            file.write(response.content)

        print(f"File downloaded successfully and saved to {file_path}")

    except Exception as e:
        print(f"Failed to download the file: {e}")


@typechecked
def upload_dataframe_to_presigned_url(
    df: pd.DataFrame,
    url: str,
    verbose: bool = False,
    check: bool = False,
) -> None:

    if check:
        csv_string = df.to_csv(index=False)
        check_dataset(csv_string)
    headers = {"Content-Type": "application/octet-stream"}

    # Create a buffer
    buffer = io.StringIO()

    # Write DataFrame to the buffer
    df.to_csv(buffer, index=False)

    # Convert buffer to string
    csv_string = buffer.getvalue()
    upload_file = json.dumps({"dataset": csv_string})
    response = requests.put(url, data=upload_file, headers=headers)
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
    # Upload a file to the specified pre-signed URL.
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
def process_dataset_response(response: dict) -> pd.DataFrame:
    # Process the response from the API into a `pandas.DataFrame`.
    if response.get("dataset") is not None:
        csv = response["dataset"]
        csv = io.StringIO(csv)
    elif response.get("dataset_url") is not None:
        url = response["dataset_url"]
        csv = download_dataframe_from_presigned_url(url)
    else:
        raise ValueError("No 'dataset' or 'dataset_url' in the response.")
    df = pd.read_csv(csv, sep=",")
    return df


def process_result_response(response: dict) -> dict:
    # Process the response from the API into a `pandas.DataFrame`.
    if response.get("result") is not None:
        result = response["result"]
    elif response.get("result_url") is not None:
        url = response["result_url"]
        result = download_result_from_presigned_url(url)
        result = json.loads(result)
    else:
        raise ValueError("No 'result' or 'result_url' in the response.")
    return result


@typechecked
def download_result_from_presigned_url(url: str) -> io.StringIO:
    # Download a `pandas.DataFrame` from the specified pre-signed URL.
    response = requests.get(url)
    result = response.json()
    return result


@typechecked
def download_dataframe_from_presigned_url(url: str) -> io.StringIO:
    # Download a `pandas.DataFrame` from the specified pre-signed URL.
    response = requests.get(url)
    buffer = io.StringIO(response.json()["dataset"])
    return buffer


@typechecked
def get_csv_string(df: pd.DataFrame) -> str:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()


@typechecked
def remove_none_values(d: dict) -> dict:
    # Remove None values from a nested dictionary.
    return {
        k: remove_none_values(v) if isinstance(v, dict) else v
        for k, v in d.items()
        if v is not None
    }


class EmulatorResultsAdapter:
    # Utility classes for munging the results of an emulator API call.

    def __init__(self, method: str, response: dict):
        self.method = method
        self.status = ValidStatus(
            response["status"]
        )  # Set the self.status as the status of the response converted to a ValidStatus
        self.result = response[
            "result"
        ]  # Set the self.result as the result of the response
        self.result_url = response[
            "result_url"
        ]  # Set the self.result_url as the result_url of the response

    def _check_status(self):
        if self.status == ValidStatus.SUCCESS:
            return True
        else:
            return False

    def _get_result(self):
        if self.result:
            # If the result is already in the response, do nothing
            pass
        elif self.result_url:
            # If a result_url is in the response, download the result
            response = requests.get(self.result_url)
            # Set the result to the downloaded result
            self.result = response.json()
        else:
            raise ValueError("No result or result_url in the response.")

    def adapt_result(self, *args, verbose: bool = False):
        # Only run if the status is success
        if self._check_status():

            self._get_result()

            # Run the adapter method for the specific method
            method = getattr(self, f"adapt_{self.method}_results")

            return method(*args, verbose=verbose)
        else:
            # What should this return?
            raise RuntimeError("Non-success status")

    def adapt_update_results(self, verbose: bool):
        return self.result["update"]

    def adapt_design_results(self, verbose: bool):
        design_csv = self.result["design"]

        design = pd.read_csv(io.StringIO(design_csv))
        return design

    def adapt_predict_results(self, verbose: bool):
        mean_csv = self.result["mean"]
        std_csv = self.result["std"]

        mean = pd.read_csv(io.StringIO(mean_csv))
        std = pd.read_csv(io.StringIO(std_csv))

        if verbose:
            print("Mean predictions:")
            pprint(mean)
            print("Standard deviation predictions:")
            pprint(std)
        return mean, std

    def adapt_sample_results(self, verbose: bool):
        sample_csv = self.result["samples"]
        sample = pd.read_csv(io.StringIO(sample_csv), header=[0, 1], sep=",")

        if verbose:
            print("Samples:")
            pprint(sample)
        return sample

    def adapt_recommend_results(self, verbose: bool):
        recommendation_csv = self.result["candidate_points"]
        acq_func_value = self.result["acq_func"]

        recommendation = pd.read_csv(io.StringIO(recommendation_csv))

        if verbose:
            print("Recommended points:")
            pprint(recommendation)
        return recommendation, acq_func_value

    def adapt_maximize_results(self, verbose: bool):
        recommendation_csv = self.result["max_point"]

        max_point = pd.read_csv(io.StringIO(recommendation_csv))

        if verbose:
            print("Maximimum point:")
            pprint(max_point)
        return max_point

    def adapt_calibrate_results(self, verbose: bool):
        calibrated_csv = self.result["calibration"]
        calibration = pd.read_csv(io.StringIO(calibrated_csv), sep=",")

        # TODO: This is a nasty hack for when return_summary=True
        if "Unnamed: 0" in calibration.columns:
            calibration.set_index("Unnamed: 0", inplace=True)
            calibration.index.name = None

        if verbose:
            print("Calibration:")
            pprint(calibration)

        return calibration

    def adapt_score_results(self, verbose: bool):
        score = self.result["score"]

        if score:
            if isinstance(score, str):
                score = pd.read_csv(io.StringIO(score))
            else:
                pass
        else:
            warnings.warn(
                "No test data was available for this emulator, so it cannot be scored."
            )

        if verbose:
            print("Emulator score:")
            pprint(score)
        return score

    def adapt_benchmark_results(self, verbose: bool):
        benchmark = self.result["calibration_curve"]

        if benchmark:
            benchmark = pd.read_csv(io.StringIO(benchmark))
        else:
            warnings.warn(
                "No test data was available for this emulator, so it cannot be scored."
            )

        if verbose:
            print("Calibration curve:")
            pprint(benchmark)
        return benchmark

    def adapt_export_results(self, file_path: str, verbose: bool):

        download_file_from_url(self.result_url, file_path)


@typechecked
def reformat_summary_dict(summary_dict: dict, detailed: bool = False) -> dict:
    # Function to reformat the summary dictionary.
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
