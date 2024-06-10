# Standard imports
import io
import json
import sys
import time
import uuid
import warnings
from pprint import pprint
from typing import Callable, Dict, List, Optional, Tuple, Union

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typeguard import typechecked

# Project imports
from . import api, settings, utils
from .dataset import Dataset
from .params import (
    BenchmarkParams,
    CalibrateParams,
    DesignParams,
    MaximizeParams,
    PredictParams,
    RecommendParams,
    SampleParams,
    ScoreParams,
    TrainParams,
)
from .plotting import DIGILAB_CMAP as digilab_cmap
from .plotting import DIGILAB_COLORS as digilab_colors
from .plotting import heatmap, plot
from .prior import Prior

# Parameters
ACQ_FUNC_DICT = {
    "EI": "EI",
    "qEI": "qEI",
    "LogEI": "LogEI",
    "qLogEI": "qLogEI",
    "PSD": "PSD",
    "qNIPV": "qNIPV",
    "ExpectedImprovement": "EI",
    "qExpectedImprovement": "qEI",
    "LogExpectedImprovement": "LogEI",
    "qLogExpectedImprovement": "qLogEI",
    "PosteriorStandardDeviation": "PSD",
    "qNegIntegratedPosteriorVariance": "qNIPV",
}
PING_TIME_INITIAL = 1.0  # Seconds
PING_FRACTIONAL_INCREASE = 0.1
PROCESSOR = "cpu"
SYNC = False
DEBUG = False
PROCESS_MAP = {
    "score": "score",
    "get_calibration_curve": "benchmark",
    "predict": "predict",
    "sample": "sample",
    "get_candidate_points": "recommend",
    "solve_inverse": "calibrate",
    "maximize": "maximize",
}

ALLOWED_DATAFRAME_SIZE = 5.5 * int(
    1e6
)  # 5.5MB limit to stay safely (also to allow other variables and dataframes to flow through without making a fuss) within the 6MB limit for a Lambda response payload size

### Helper functions ###
# TODO: Should these functions all have preceeding underscores?


@typechecked
def _upload_large_datasets(
    df: pd.DataFrame, csv_string: str, method_prefix: str
) -> Optional[str]:
    df_id = str(uuid.uuid4())
    if sys.getsizeof(csv_string) > ALLOWED_DATAFRAME_SIZE:
        dataset_id = method_prefix + "_data_" + df_id
        _, response = api.generate_temp_upload_url(dataset_id, verbose=DEBUG)
        upload_url = utils.get_value_from_body("url", response)
        utils.upload_dataframe_to_presigned_url(
            df,
            upload_url,
            check=settings.CHECK_DATASETS,
        )
        return dataset_id
    else:
        return None


@typechecked
def _retrieve_dataframe_from_response(
    response: dict,
    dataframe_key: Optional[str] = "dataframe",
    dataframe_url_key: Optional[str] = "dataframe_url",
) -> io.StringIO:
    if response[dataframe_key] is not None:
        csv = utils.get_value_from_body(dataframe_key, response)
        csv = io.StringIO(csv)
    else:
        data_url = utils.get_value_from_body(dataframe_url_key, response)
        csv = utils.download_dataframe_from_presigned_url(data_url)
    return csv


@typechecked
def _calculate_ping_time(elapsed_time: float) -> float:
    # This smoothly transitions between regular pinging at the initial ping time
    # to more drawn out pinging (expoential) as time goes on
    return PING_TIME_INITIAL + elapsed_time * PING_FRACTIONAL_INCREASE


# TODO: Combine _wait_for_training_completion and _wait_for_job_completion
@typechecked
def _wait_for_training_completion(
    model_id: str, process_id: str, verbose: bool = False
) -> None:
    start_time = time.time()
    status = 202
    while status == 202:
        elapsed_time = time.time() - start_time  # This will be ~0 seconds initially
        wait_time = _calculate_ping_time(elapsed_time)
        time.sleep(wait_time)
        status, body = api.train_response_model(
            model_id=model_id,
            process_id=process_id,
            verbose=DEBUG,
        )
        if verbose:
            message = _get_response_message(body)
            print(f"Training status: {message}")


# TODO: Combine _wait_for_training_completion and _wait_for_job_completion
@typechecked
def _wait_for_job_completion(
    model_id: str, method: str, process_id: str, verbose: bool = False
) -> Tuple[int, dict]:
    start_time = time.time()
    status = 202
    while status == 202:
        elapsed_time = time.time() - start_time  # This will be ~0 seconds initially
        wait_time = _calculate_ping_time(elapsed_time)
        time.sleep(wait_time)
        status, body = api.use_response_model(
            model_id=model_id,
            method=method,
            process_id=process_id,
            verbose=DEBUG,
        )
        if verbose:
            message = _get_response_message(body)
            print(f"Job status for {PROCESS_MAP[method]}: {message}")
    return status, body


# TODO: This function needs streamlining, what about dict.get(key, default)?
# TODO: All responses should return a "message", then this would not be necessary
@typechecked
def _get_response_message(body: dict) -> str:
    if "message" in body:  # TODO: if/else structure needs to be streamlined
        message = body["message"]
    elif "process_status" in body:
        message = body["process_status"]
    elif (
        "process_status:" in body
    ):  # TODO: Note the colon; this needs to be made more robust
        message = body["process_status:"]
    else:
        message = "No response message in body"
    return message


@typechecked
def _process_csv(
    csv: io.StringIO, method: str
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    if method == "predict":
        df = pd.read_csv(csv, sep=",")
        n = len(df.columns)
        df_mean, df_std = df.iloc[:, : n // 2], df.iloc[:, n // 2 :]
        df_std.columns = df_std.columns.str.removesuffix(" [std_dev]")
        return df_mean, df_std
    elif method == "maximize":
        df = pd.read_csv(csv, sep=",")
        return df
    elif method == "sample":
        df_result = pd.read_csv(csv, header=[0, 1], sep=",")
        return df_result
    elif method == "get_candidate_points":
        df = pd.read_csv(csv, sep=",")
        return df
    elif method == "solve_inverse":
        df = pd.read_csv(csv, sep=",")
        df = df.set_index("Unnamed: 0")
        df.index.name = None
        if "Unnamed: 0.1" in df.columns:  # TODO: This seems like a nasty hack
            df = df.drop("Unnamed: 0.1", axis=1)
        return df
    else:
        raise ValueError(f"Method {method} not recognised")


### ###


class Emulator:
    """A trainable twinLab emulator.

    An emulator is trainable model that learns the trends in a dataset.
    It is a machine-learning model in that it requires a dataset of inputs ``X`` and outputs ``y`` on which to be trained.
    In this way, it learns to mimic, or emulate, the dataset and can be used to make predictions on new data.
    Emulators are also often called models, surrogates, or digital twins.

    Note that instantiating an emulator locally does not create a new emulator on the twinLab cloud.
    Instead, it can be used either to interact with an existing emulator that has previously been trained, or as a precursor step to training a new emulator.

    Attributes:
        id (str): The name for the emulator in the twinLab cloud.
            If an emulator is specified that does not currently exist, then a new emulator will be instantiated.
            Otherwise the corresponding emulator will be loaded from the cloud.
            Be sure to double check which emulators have been created using ``tl.list_emulators``.
    """

    @typechecked
    def __init__(self, id: str):
        self.id = id

    @typechecked
    def design(
        self,
        priors: List[Prior],
        num_points: int,
        params: DesignParams = DesignParams(),
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Generate an initial design space for an emulator.

        The method is used to generate an initial design for evaluating a set of experiments, emulator, or simulation.
        This is useful if data has not yet been collected and a user wants to generate an initial design space to train an emulator.
        Optimal space-filling methods can be used to generate an initial design space that are significantly better than either random or grid sampling.
        If data has already been acquired then an initial emulator can be trained using ``Emulator.train()`` and new sampling locations can be recommended using ``Emulator.recommend()``.

        Args:
            priors (list[Prior]): A list of ``Prior`` objects that define the prior distributions for each input.
                These are independent one-dimensional probability distributions for each parameter.
            num_points (int): The number of points to sample in designing the initial space.
            params (twinlab.DesignParams, optional): A parameter configuration that contains all of the optional initial-design parameters.

        Example:
            .. code-block:: python

                emulator = tl.Emulator("emulator_id")

                my_priors = [
                    tl.Prior("x1", tl.distributions.Uniform(0, 12)),
                    tl.Prior("x2", tl.distributions.Uniform(0, 0.5)),
                    tl.Prior("x3 ", tl.distributions.Uniform(0, 10)),
                ]

                initial_design = emulator.design(my_priors, 10)

        """
        # Convert priors to json so they can be passed through the API
        serialised_priors = [prior.to_json() for prior in priors]

        # Call the API function
        _, response = api.get_initial_design(
            serialised_priors,
            params.sampling_method.to_json(),
            num_points,
            seed=params.seed,
            verbose=verbose,
        )
        # Get result from body of response
        initial_design = _retrieve_dataframe_from_response(
            response, "initial_design", "initial_design_url"
        )
        if (
            "dataframe_name" in response.keys()
            and response["dataframe_name"] is not None
        ):
            api.delete_temp_dataset(response["dataframe_name"])

        # Convert result which is a numpy array to pandas dataframe with correct column names
        initial_design_df = pd.read_csv(initial_design, sep=",")

        if verbose:
            print("Initial design:")
            print(initial_design_df)

        return initial_design_df

    @typechecked
    def train(
        self,
        dataset: Dataset,
        inputs: List[str],
        outputs: List[str],
        params: TrainParams = TrainParams(),
        wait: bool = True,
        verbose: bool = True,
    ) -> Optional[str]:
        """Train an emulator on the twinLab cloud.

        This is the primary functionality of twinLab, where an emulator is trained on a dataset.
        The emulator learns trends in the dataset and then is able to make predictions on new data.
        These new data can be far away from the training data, and the emulator will interpolate between the training data points.
        The emulator can also be used to extrapolate beyond the training data, but this is less reliable.
        The emulator can be trained on a dataset with multiple inputs and outputs,
        and can be used to make predictions on new data with multiple inputs and outputs.
        The powerful algorithms in twinLab allow for the emulator to not only make predictions,
        but to also quantify the uncertainty in these predictions.
        This is extremely advantageous, because it allows for the reliability of the predictions to be quantified.

        Args:
            dataset (Dataset): The training and test data for the emulator.
                The ratio of train to test data can be set in ``TrainParams``.
            inputs (list[str]): A list of the input column names in the training dataset.
                These correspond to the independent variables in the dataset, which are often the parameters of a model.
                These are usually known as ``X`` (note that capital) values.
            outputs (list[str]): A list of the output column names in the training dataset.
                These correspond to the dependent variables in the dataset, which are often the results of a model.
                These are usually known as ``y`` values.
            params (TrainParams, optional): A training parameter configuration that contains all optional training parameters.
            wait (bool, optional): If ``True`` wait for the job to complete, otherwise return the process ID and exit.
                Setting ``wait=False`` is useful for running longer training jobs.
                The status of all emulators, including those currently training, can be queried using ``tl.list_emulators(verbose=True)``.
            verbose (bool, optional): Display information about the operation while running.

        Returns:
            If ``wait=True`` the function will run until the emulator is trained on the cloud.
            If ``wait=False`` the function will return the process ID and exit.
            This is useful for longer training jobs.
            The training status can then be checked later using ``Emulator.status()``.

        Example:
            .. code-block:: python

                df = pd.DataFrame({"X": [1, 2, 3, 4], "y": [1, 4, 9, 16]})
                dataset = tl.Dataset("my_dataset")
                dataset.upload(df)
                emulator = tl.Emulator("my_emulator")
                emulator.train(dataset, ["X"], ["y"])

        """

        # Making a dictionary from TrainParams class
        if PROCESSOR == "gpu":
            print(
                "Emulator is being trained on GPU. Inference operations must also be performed on GPU"
            )
        train_dict = params.unpack_parameters()
        train_dict["inputs"] = inputs
        train_dict["outputs"] = outputs
        train_dict["dataset_id"] = dataset.id
        train_dict = utils.coerce_params_dict(train_dict)
        params_str = json.dumps(train_dict)

        # Send training request
        _, response = api.train_request_model(
            self.id, params_str, processor=PROCESSOR, verbose=DEBUG
        )
        if verbose:
            message = utils.get_message(response)
            print(message)

        # Get the process ID from the response
        process_id = utils.get_value_from_body("process_id", response)
        if verbose:
            print(f"Emulator {self.id} with process ID {process_id} is training.")
        if not wait:
            return process_id
        _wait_for_training_completion(self.id, process_id, verbose=verbose)
        if verbose:
            print(
                f"Training of emulator {self.id} with process ID {process_id} is complete!"
            )

    @typechecked
    def status(self, process_id: str, verbose: bool = False) -> dict:
        """Check the status of a training process on the twinLab cloud.

        Args:
            process_id (str): The process ID of the training process to check the status of.
            verbose (bool, optional): Display information about the operation while running.

        Returns:
            Tuple[int, dict]: A tuple containing the status code and the response body.

        Example:
            .. code-block:: python

                emulator = tl.Emulator("beb7f97f")
                emulator.status()

            .. code-block:: console

                {
                    'process_status': 'Your job has finished and is on its way back to you.',
                    'process_id': 'beb7f97f',
                }

        """
        _, response = api.train_response_model(self.id, process_id, verbose=DEBUG)
        message = _get_response_message(response)
        if verbose:
            print(message)
        return response

    @typechecked
    def view(self, verbose: bool = False) -> dict:
        """View an emulator that exists on the twinLab cloud.

        This returns the parameter configuration of the emulator that is stored on the twinLab cloud.
        This allows a user to check the parameters that were used to train an emulator.

        Args:
            verbose (bool, optional): Display information about the operation while running.

        Example:
            .. code-block:: python

                emulator = tl.Emulator("quickstart")
                emulator.view()

            .. code-block:: console

                {'dataset_id': 'quickstart',
                 'decompose_inputs': False,
                 'decompose_outputs': False,
                 'estimator': 'gaussian_process_regression',
                 'estimator_kwargs': {'detrend': False, 'estimator_type': 'single_task_gp'},
                 'inputs': ['x'],
                 'modal_handle': 'fc-6L9EsWZhOkc8xyHguPphh6',
                 'model_id': 'quickstart',
                 'model_selection': False,
                 'model_selection_kwargs': {'base_kernels': 'restricted',
                                            'depth': 1,
                                            'evaluation_metric': 'MSLL',
                                            'val_ratio': 0.2},
                 'outputs': ['y'],
                 'train_test_ratio': 0.8}

        """

        _, response = api.view_model(self.id, verbose=DEBUG)
        parameters = (
            response  # Note that the whole body of the response is the parameters
        )
        if verbose:
            print("Emulator parameters summary:")
            pprint(parameters, compact=True, sort_dicts=False)
        return parameters

    @typechecked
    def view_train_data(self, verbose: bool = False) -> pd.DataFrame:
        """View training data with which the emulator was trained in the twinLab cloud.

        Args:
            verbose (bool, optional): Display information about the operation while running.

        Returns:
            pandas.DataFrame: Dataframe containing the training data on which the emulator was trained

        Example:
            .. code-block:: python

                emulator = tl.Emulator("quickstart")
                emulator.view_train_data()

            .. code-block:: console

                          x         y
                0  0.696469 -0.817374
                1  0.286139  0.887656
                2  0.226851  0.921553
                3  0.551315 -0.326334
                4  0.719469 -0.832518
                5  0.423106  0.400669
                6  0.980764 -0.164966
                7  0.684830 -0.960764

        """
        _, response = api.view_data_model(self.id, dataset_type="train", verbose=DEBUG)
        train_csv_string = _retrieve_dataframe_from_response(
            response, "training_data", "training_data_url"
        )
        df_train = pd.read_csv(train_csv_string, sep=",", index_col=0)
        if verbose:
            print("Training data")
            pprint(df_train)
        return df_train

    @typechecked
    def view_test_data(self, verbose: bool = False) -> pd.DataFrame:
        """View test data on which the emulator was tested in the twinLab cloud.

        Args:
            verbose (bool, optional): Display information about the operation while running.

        Returns:
            pandas.DataFrame: Dataframe containing the training data on which the emulator was tested.

        Example:

            .. code-block:: python

                emulator = tl.Emulator("quickstart")
                emulator.view_test_data()

            .. code-block:: console

                          x         y
                0  0.480932  0.340115
                1  0.392118  0.845795

        """
        _, response = api.view_data_model(self.id, dataset_type="test", verbose=DEBUG)
        test_csv_string = _retrieve_dataframe_from_response(
            response, "test_data", "test_data_url"
        )
        df_test = pd.read_csv(test_csv_string, sep=",", index_col=0)
        if verbose:
            print("Test data")
            pprint(df_test)
        return df_test

    def list_processes(self, verbose: bool = False) -> Dict[str, Dict]:
        """List all of the processes associated with a given emulator on the twinLab cloud.

        Args:
            verbose (bool, optional): Determining level of information returned to the user. Default is False.

        Returns:
            dict: Dictionary containing all processes associated with the emulator.

        Example:

            .. code-block:: python

                emulator = tl.Emulator("quickstart")
                emulator.list_processes()

            .. code-block:: console

                [
                    {
                        'method': 'sample',
                        'process_id': '23346a9c',
                        'run_time': '0:00:05',
                        'start_time': '2024-04-09 17:10:12',
                        'status': 'success'
                    },
                    {
                        'method': 'sample',
                        'process_id': '676623b0',
                        'run_time': '0:00:04',
                        'start_time': '2024-04-09 18:45:48',
                        'status': 'success'
                    },
                ]

        """
        _, response = api.list_processes_model(model_id=self.id, verbose=DEBUG)
        processes = utils.get_value_from_body("processes", response)

        # Create dictionary of cuddly response
        status_dict = {
            "success": "Successful processes:",
            "in_progress": "Currently running processes:",
            "failed": "Processes that failed to complete:",
        }

        verbose_keys = ("method", "start_time", "run_time")

        if verbose:
            if not processes:
                print("No processes available for this emulator.")
            for status, nice_status in status_dict.items():
                procs = [proc for proc in processes if proc["status"] == status]
                # Sort through dictionary via success, in_progress, failed
                procs = [
                    dict((key, proc[key]) for key in verbose_keys) for proc in procs
                ]
                procs = sorted(procs, key=lambda d: d["start_time"])
                # List models in order from starting time
                if procs:
                    # Only print list if there are available processes in the list
                    print(nice_status)
                    pprint(procs)
        return processes

    @typechecked
    def get_process(
        self, process_id: str, verbose: bool = False
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Get the results from a process associated with the emulator on the twinLab cloud.

        This allows a user to retrieve any results from processes (jobs) they have run previously.
        The list of available process IDs can be obtained from the ``list_processes()`` method.

        Args:
            process_id (str): The ID of the process from which to get the results.
            verbose (bool, optional): Display information about the operation while running.

        Example:
            .. code-block:: python

                emulator = tl.Emulator("quickstart")
                emulator.get_process("23346a9c")

            .. code-block:: console

                            y
                            0         1         2         3
                0   -0.730114  0.474193  0.046743  1.327620
                1   -0.656061  0.505923  0.074198  1.289113
                2   -0.579500  0.538610  0.100665  1.247405
                3   -0.502726  0.574996  0.128068  1.205057
                4   -0.428691  0.614687  0.157740  1.165903

        """
        method = " "  # Error generated regarding "local variable referenced before asssignment" if not included
        _, response = api.list_processes_model(model_id=self.id, verbose=DEBUG)
        method = " "  # TODO: This is a bad way to avoid a scope error for this variable

        for i in range(len(response["processes"])):
            if response["processes"][i]["process_id"] == process_id:
                method = response["processes"][i]["method"]
        _, response = api.use_response_model(
            model_id=self.id,
            method=method,
            process_id=process_id,
            verbose=DEBUG,
        )

        csv = _retrieve_dataframe_from_response(response)
        if method == "predict":
            df_mean, df_std = _process_csv(csv, method)
            if verbose:
                print("Mean predictions:")
                print(df_mean)
                print("Standard deviation predictions:")
                print(df_std)
            return df_mean, df_std
        else:
            df = _process_csv(csv, method)
            if verbose:
                if method == "sample":
                    print("Samples:")
                elif method == "get_candidate_points":
                    print("Recommended points:")
                elif method == "solve_inverse":
                    print("Calibration summary:")
                else:
                    print("Process results:")
                print(df)
            return df

    @typechecked
    def summarise(self, detailed: bool = False, verbose: bool = False) -> dict:
        """Get a summary of a trained emulator on the twinLab cloud.

        This summary returns transformer diagnostics, with details about the input/output decomposition.
        It also returns the estimator diagnostics, detailing properties of the trained emulator.
        The estimator diagnostics includes information about the kernel, likelihood, mean modules and some additional properties of the emulator.
        This information can help inform a user about the makeup of an emulator -- for example, what kind of kernel was used.
        The summary contains the information about the following components of an emulator.

        +----------------------------+----------------------------------------------------------------------------------+
        |           Kernel           | Details about the structure of the kernel used, kernel lengthscale priors,       |
        |                            | kernel lengthscale bounds, etc.                                                  |
        +----------------------------+----------------------------------------------------------------------------------+
        |         Likelihood         | Details about the noise model used (could be another GP in the case of           |
        |                            | Heteroskedastic GP), raw noise variance, etc.                                    |
        +----------------------------+----------------------------------------------------------------------------------+
        |            Mean            | Details about the mean function used and the raw mean constant.                  |
        +----------------------------+----------------------------------------------------------------------------------+
        |                            | Additional properties of the emulator like noise variance, input transform       |
        |         Properties         | parameters, output transform parameters, properties of the variational           |
        |                            | distribution (only in the case of Variational GPs), etc.                         |
        +----------------------------+----------------------------------------------------------------------------------+

        By default, the summary dictionary is not detailed and contains only a high-level overview of the emulator which includes information about
        the learned hyperparameters, kernel function used and the mean function used. To view a detailed summary innvolving information about all the
        parameters of the emulator, set the ``detailed`` parameter to ``True``.

        Args:
            verbose (bool, optional): Display information about the operation while running.

        Example:
            .. code-block:: python

                emulator = tl.Emulator("quickstart")
                emulator.summarise()

            .. code-block:: console

                {
                    'kernel': ...,
                    'likelihood': ...,
                    'mean': ...,
                    'properties': ...,
                }

            .. code-block:: python

                emulator = tl.Emulator("quickstart")
                emulator_summary = emulator.summarise()
                kernel_info = emulator_summary.get("kernel")
                print("Kernel function used: ", kernel_info["kernel_function_used"])
                print("Lengthscale:", kernel_info["lengthscale"])

            .. code-block:: console

                Kernel function used: ScaleKernel((base_kernel): MaternKernel((lengthscale_prior): GammaPrior()
                (raw_lengthscale_constraint): Positive())
                (outputscale_prior): GammaPrior()  (raw_outputscale_constraint): Positive())

                Lengthscale: [[0.4234508763532827]]


        """
        _, response = api.summarise_model(self.id, verbose=DEBUG)
        summary = utils.get_value_from_body("model_summary", response)
        del summary["data_diagnostics"]
        if "base_estimator_diagnostics" in summary["estimator_diagnostics"].keys():
            base_estimator_summary = utils.reformat_summary_dict(
                summary, detailed=detailed
            )
            summary["estimator_diagnostics"][
                "base_estimator_diagnostics"
            ] = base_estimator_summary
        else:
            summary = utils.reformat_summary_dict(summary, detailed=detailed)
        if verbose:
            print("Trained emulator summary:")
            pprint(summary, compact=True, sort_dicts=False)
        return summary

    @typechecked
    def _use_method(
        self,
        method: str,
        df: Optional[pd.DataFrame] = None,
        df_std: Optional[pd.DataFrame] = None,
        verbose: bool = False,
        **kwargs,  # NOTE: This can be *anything*
    ):
        if df is not None:
            data_csv = df.to_csv(index=False)
        else:
            data_csv = None
        if df_std is not None:
            data_std_csv = df_std.to_csv(index=False)
        else:
            data_std_csv = None
        _, response = api.use_model(
            self.id,
            method,
            data_csv=data_csv,
            data_std_csv=data_std_csv,
            **kwargs,
            processor=PROCESSOR,
            verbose=DEBUG,
        )

        # Check if an acq func value exists in the response
        # If it does then also return it with the dataframe
        if "acq_func_value" in response.keys():
            acq_func_value = utils.get_value_from_body("acq_func_value", response)
            return io.StringIO(output_csv), acq_func_value
        if "dataframe" in response.keys():
            output_csv = utils.get_value_from_body("dataframe", response)
            return io.StringIO(output_csv)
        else:
            output = utils.get_value_from_body("result", response)
            return output

    @typechecked
    def score(
        self,
        params: ScoreParams = ScoreParams(),
        verbose: bool = False,
    ) -> Optional[Union[pd.DataFrame, float]]:
        """Score the performance of a trained emulator.

        Returns a score for a trained emulator that quantifies its performance on the test dataset.
        Note that a test dataset must have been defined in order for this to produce a result.
        This means that ``train_test_ratio`` in TrainParams must be less than ``1`` when training the emulator.
        If there is no test dataset then this will return ``None``.
        The score can be calculated using different metrics, see the ``ScoreParams`` class for a full list and description of available metrics.

        Args:
            params (ScoreParams, optional): A parameters object that contains optional scoring parameters.
            verbose (bool, optional): Display detailed information about the operation while running.

        Returns:
            Either a ``pandas.DataFrame`` containing the emulator per output dimension (if ``combined_score = False``),
            or a ``float`` containing the combined score of the emulator averaged acorss output dimensions (if ``combined_score = True``),
            or ``None`` if there was no test data defined during training.

        Examples:

            Request the mean-standarised log loss (MSLL) averaged (combined) across all emulator output dimensions:

            .. code-block:: python

                emulator = tl.Emulator("my_emulator")
                params = tl.ScoreParams(metric="MSLL", combined_score=True)
                emulator.score(params=params)

            .. code-block:: console

                -4.07

            Request the mean-squared error (MSE) for each output individually:

            .. code-block:: python

                emulator = tl.Emulator("my_emulator")
                params = tl.ScoreParams(metric="MSE", combined_score=False)
                emulator.score(params=params)

            .. code-block:: console

                pd.DataFrame({'y1': [1.8], 'y2': [0.9]})
        """

        score = self._use_method(
            method="score",
            **params.unpack_parameters(),
            verbose=verbose,
        )

        # Only return the score if there is test data
        if score is not None:
            if not params.combined_score:  # DataFrame
                score = pd.read_csv(score, sep=",")
            if verbose:
                print("Emulator Score:")
                print(score)  # Could be pd.DataFrame or float
            return score
        else:
            warnings.warn(
                "No test data was available for this emulator, so it cannot be scored."
            )

    @typechecked
    def benchmark(
        self,
        params: BenchmarkParams = BenchmarkParams(),
        verbose: bool = False,
    ) -> Optional[pd.DataFrame]:
        """Benchmark the predicted uncertainty of a trained emulator.

        A test dataset must have been defined in order for this to produce a result (otherwise ``None`` is returned).
        This means that ``train_test_ratio`` must be less than 1 when training the emulator.
        This method returns the calibration curve of a trained emulator, which can be used to asses the quality of the uncertainty predictions.
        The calibration curve is a plot of the fraction of values that are predicted to be within a certain confidence interval against the actual fraction of values that are within that interval.

        100 monotonically increasing values between 0 and 1 are returned for each output dimension of the emulator, in the form of a ``pandas.DataFrame``.
        These values can be plotted as the values on a y-axis, while the x-axis is taken to be 100 equally spaced values between 0 and 1 (inclusive).
        A well-calibrated emulator will have a curve that is close to the line ``y = x``.
        If the shape deviates from this line, the emulator may be under- or overconfident, but the exact interpretation depends on the type of curve.
        See the documentation for ``BenchmarkParams`` for more information on the available types.

        Args:
            params (BenchmarkParams, optional): A parameter-configuration object that contains optional parameters for benchmarking an emulator.
            verbose (bool, optional): Display detailed information about the operation while running.

        Returns:
            pandas.DataFrame, None: Either a ``pandas.DataFrame`` containing the calibration curve for an emulator, or ``None`` if there is no test data.

        Example:
            .. code-block:: python

                emulator = tl.Emulator("quickstart")
                emulator.benchmark()

            .. code-block:: console

                      y
                0   0.0
                1   0.0
                2   0.0
                3   0.0
                4   0.0
                ..  ...
                95  1.0
                96  1.0
                97  1.0
                98  1.0
                99  1.0

        """
        # TODO: Maybe include how to plot calibration curve in a docstring code snippet.

        csv = self._use_method(
            method="get_calibration_curve",
            **params.unpack_parameters(),
            verbose=verbose,
        )

        # Only return the DataFrame if there is test data
        if csv is not None:
            df = pd.read_csv(csv, sep=",")
            if verbose:
                print("Calibration curve:")
                pprint(df)
            return df
        else:
            warnings.warn(
                "No test data was available for this emulator, so it cannot be scored."
            )

    @typechecked
    def predict(
        self,
        df: pd.DataFrame,
        params: PredictParams = PredictParams(),
        wait: bool = True,
        verbose: bool = True,
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], str]:
        """Make predictions using a trained emulator that exists on the twinLab cloud.

        This method makes predictions from a trained emulator on new data.
        This method is the workhorse of the twinLab suite, allowing users to make predictions based on their training data.
        The emulator can make predictions on data that are far away from the training data, and can interpolate reliably between the training data points.
        The emulator returns both a predicted mean and standard deviation for each output dimension.
        This allows a user to not only make predictions, but also to quantify the uncertainty on those predictions.
        For a Gaussian Process, the standard deviation is a measure of the uncertainty in the prediction,
        while the mean is the prediction itself.
        The emulator is 95% confident that the true value lies within two standard deviations of the mean.

        Args:
            df (pandas.DataFrame): The ``X`` values for which to make predictions.
            params (PredictParams): A parameter-configuration that contains optional parameters for making predictions.
            wait (bool, optional): If ``True`` wait for the job to complete, otherwise return the process ID and exit.
            verbose (bool, optional): Display detailed information about the operation while running.

        Returns:
            Tuple[pandas.DataFrame, pandas.DataFrame], str: By default a tuple containing the mean and standard deviation of the emulator prediction.
            Instead, if ``wait=False``, the process ID is returned.
            The results can then be retrieved later using ``Emulator.get_process(<process_id>)``.
            Process IDs associated with an emulator can be found using ``Emulator.list_processes()``.

        Example:
            .. code-block:: python

                emulator = tl.Emulator("quickstart")
                df = pd.DataFrame({'x': [0.1, 0.2, 0.3, 0.4]})
                df_mean, df_std = emulator.predict(df)

            .. code-block:: console

                          y
                0  0.845942
                1  0.922921
                2  0.846308
                3  0.570473

                          y
                0  0.404200
                1  0.180853
                2  0.146619
                3  0.147886

        """
        API_METHOD = "predict"
        if SYNC:
            csv = self._use_method(
                method=API_METHOD,
                df=df,
                **params.unpack_parameters(),
                verbose=verbose,
            )
        else:
            data_csv = utils.get_csv_string(df)
            dataset_id = _upload_large_datasets(df, data_csv, API_METHOD)
            if dataset_id is not None:
                data_csv = None
            _, response = api.use_request_model(
                model_id=self.id,
                method=API_METHOD,
                dataset_id=dataset_id,
                data_csv=data_csv,
                **params.unpack_parameters(),
                processor=PROCESSOR,
                verbose=DEBUG,
            )
            process_id = utils.get_value_from_body("process_id", response)
            if verbose:
                print(f"Job {PROCESS_MAP[API_METHOD]} process ID: {process_id}")
            if not wait:
                return process_id
            _, response = _wait_for_job_completion(
                self.id, API_METHOD, process_id, verbose=verbose
            )
            csv = _retrieve_dataframe_from_response(response)
        df_mean, df_std = _process_csv(csv, API_METHOD)
        if verbose:
            print("Mean predictions:")
            print(df_mean)
            print("Standard deviation predictions:")
            print(df_std)
        return df_mean, df_std

    @typechecked
    def sample(
        self,
        df: pd.DataFrame,
        num_samples: int,
        params: SampleParams = SampleParams(),
        wait: bool = True,
        verbose: bool = True,
    ) -> Union[pd.DataFrame, str]:
        """Draw samples from a trained emulator that exists on the twinLab cloud.

        A secondary functionality of the emulator is to draw sample predictions from the trained emulator.
        Rather than quantifying the uncertainty in the predictions, this method draws samples from the emulator.
        The collection of samples can be used to explore the distribution of the emulator predictions.
        Each sample is a possible prediction of the emulator, and therefore a prediction of a possible new observation from the data-generation process.
        The covariance in the emulator predictions can therefore be explored, which is particularly useful for functional Gaussian Processes.

        If the output of the multi-indexed DataFrame needs to be manipulated then we provide the convenience functions:

        - ``tl.get_sample``: Isolate an individual sample into a new ``pandas.DataFrame``
        - ``tl.join_samples``: Join together multiple sets of samples into a single ``pandas.DataFrame``

        Args:
            df (pandas.DataFrame): The ``X`` values for which to draw samples.
            num_samples (int): Number of samples to draw for each row of the evaluation data.
            params (SampleParams, optional): A `SampleParams` object with sampling parameters.
            wait (bool, optional): If ``True`` wait for the job to complete, otherwise return the process ID and exit.
            verbose (bool, optional): Display detailed information about the operation while running.

        Returns:
            pandas.DataFrame, str: By default a multi-indexed DataFrame containing the ``y`` samples drawn from the emulator.
            Instead, if ``wait=False`` the process ID is returned.
            The results can then be retrieved later using ``Emulator.get_process(<process_id>)``.
            Process IDs associated with an emulator can be found using ``Emulator.list_processes()``.

        Example:
            .. code-block:: python

                emulator = tl.Emulator("quickstart")
                df = pd.DataFrame({'x': [0.1, 0.2, 0.3, 0.4]})
                emulator.sample(df, 3)

            .. code-block:: console

                          y
                          0         1         2
                0  0.490081  1.336099  0.608441
                1  0.829179  1.038671  0.807405
                2  0.805102  0.773975  0.984713
                3  0.605568  0.416630  0.713652

        """
        API_METHOD = "sample"
        if SYNC:
            csv = self._use_method(
                method=API_METHOD,
                df=df,
                num_samples=num_samples,
                **params.unpack_parameters(),
                verbose=verbose,
            )
        else:
            data_csv = utils.get_csv_string(df)
            dataset_id = _upload_large_datasets(df, data_csv, API_METHOD)
            if dataset_id is not None:
                data_csv = None
            _, response = api.use_request_model(
                model_id=self.id,
                method=API_METHOD,
                dataset_id=dataset_id,
                data_csv=data_csv,
                num_samples=num_samples,
                **params.unpack_parameters(),
                processor=PROCESSOR,
                verbose=DEBUG,
            )
            process_id = utils.get_value_from_body("process_id", response)
            if verbose:
                print(f"Job {PROCESS_MAP[API_METHOD]} process ID: {process_id}")
            if not wait:
                return process_id
            _, response = _wait_for_job_completion(
                self.id, API_METHOD, process_id, verbose=verbose
            )
            csv = _retrieve_dataframe_from_response(response)
        df = _process_csv(csv, API_METHOD)
        if verbose:
            print("Samples:")
            print(df)
        return df

    @typechecked
    def recommend(
        self,
        num_points: int,
        acq_func: str,
        params: RecommendParams = RecommendParams(),
        wait: bool = True,
        verbose: bool = True,
    ) -> Union[Tuple[pd.DataFrame, float], str]:
        """Draw new recommended data points from a trained emulator that exists on the twinLab cloud.

        The recommend functionality of an emulator is used to suggest new data points to sample.
        These new data points can be chosen depending on a variety of different user objectives.
        Currently, the user can choose between ``"optimise"`` and ``"explore"`` acquisition functions:

        - ``"optimise"`` will obtain suggested ``"X"`` values the evaluation of which (acquiring the corresponding ``"y"``) will maximize the knowledge of the emulator about the location of the maximum.
          A classic use case for this would be a user trying to maximize the output of a model (e.g., the combination of metals that creates the strongest alloy).
          This method can also be used to minimize an output, by using the ``weight`` argument of ``RecommendParams`` to multiply the output by -1.
          If an emulator has more-than-one output, then a weighted combination of the outputs can be minimized/maximized.
          Once again, using the ``opt_weight`` argument of ``MaximizeParams`` can control the weight assigned to each output, or can be used to focus on a single output.
          For example, the maximum strength of a pipe given a set of design parameters.
        - ``"explore"`` will instead suggest ``"X"`` that reduce the overall uncertainty of the emulator across the entire input space.
          A classic use case for this would be a user trying to reduce overally uncertainty.
          For example, a user trying to reduce the uncertainty in the strength of a pipe across all design parameters.

        The number of requested data points can be specified by the user, and if this is greater than 1 then then recommendations are all suggested at once, and are designed to be the optmial set, as a group, to achieve the user outcome.
        twinLab optimises which specific acquisition function within the chosen category will be used, prioritising numerical stability based on the number of points requested.

        The value of the acquisition function is also returned to the user.
        While this is of limited value in isolation, the trend of the acquisition function value over multiple iterations of ``Recommend`` can be used to understand the performance of the emulator.
        The ``Emulator.learn`` method can be used to improve the performance of an emulator iteratively.

        Args:
            num_points (int): The number of samples to draw for each row of the evaluation data.
            acq_func (str): Specifies the acquisition function to be used when recommending new points.
                The acquisition function can be either ``"explore"`` or ``"optimise"``.
            params (RecommendParams, optional): A parameter configuration that contains all of the optional recommendation parameters.
            wait (bool, optional): If ``True`` wait for the job to complete, otherwise return the process ID and exit.
            verbose (bool, optional): Display detailed information about the operation while running.

        Returns:
            Tuple[pandas.DataFrame, float], str: By default, a tuple is returned containing the recommended samples and the acquisition function value.
            Instead, if ``wait=False``, the process ID is returned.
            The results can then be retrieved later using ``Emulator.get_process(<process_id>)``.
            Process IDs associated with an emulator can be found using ``Emulator.list_processes()``.

        Example:
            .. code-block:: python

                emulator = tl.Emulator("quickstart")
                emulator.recommend(5, "explore")

            .. code-block:: console

                          x
                0  0.852853
                1  0.914091
                2  0.804012
                3  0.353463
                4  0.595268

                -0.00553509

        Example:

            .. code-block:: python

                emulator = tl.Emulator("quickstart")
                emulator.recommend(3, "optimise")

            .. code-block:: console

                          x
                0  0.273920
                1  0.306423
                2  0.226851

                0.046944751

        """
        API_METHOD = "get_candidate_points"

        # Convert acq_func names to correct method depending on number of points requested
        if acq_func == "optimise":
            if num_points == 1:
                acq_func = "EI"
            else:
                acq_func = "qEI"
        if acq_func == "explore":
            acq_func = "qNIPV"

        if SYNC:
            csv, acq_func_value = self._use_method(
                method=API_METHOD,
                num_points=num_points,
                acq_func=ACQ_FUNC_DICT[acq_func],
                **params.unpack_parameters(),
                verbose=verbose,
            )

        else:
            _, response = api.use_request_model(
                model_id=self.id,
                method=API_METHOD,
                num_points=num_points,
                acq_func=ACQ_FUNC_DICT[acq_func],
                **params.unpack_parameters(),
                processor=PROCESSOR,
                verbose=DEBUG,
            )
            process_id = utils.get_value_from_body("process_id", response)
            if verbose:
                print(f"Job {PROCESS_MAP[API_METHOD]} process ID: {process_id}")
            if not wait:
                return process_id
            _, response = _wait_for_job_completion(
                self.id, API_METHOD, process_id, verbose=verbose
            )
            csv = _retrieve_dataframe_from_response(response)
            acq_func_value = float(
                utils.get_value_from_body("acq_func_value", response)
            )

        df = _process_csv(csv, API_METHOD)
        if verbose:
            print("Recommended points:")
            print(df)
            print("Acquisition function value:")
            print(acq_func_value)
        return df, acq_func_value

    @typechecked
    def calibrate(
        self,
        df_obs: pd.DataFrame,
        df_std: pd.DataFrame,
        params: CalibrateParams = CalibrateParams(),
        wait: bool = True,
        verbose: bool = True,
    ) -> Union[pd.DataFrame, str]:
        """Solve an inverse problem using a trained emulator on the twinLab cloud.

        A classic trained emulator can ingest ``X`` values and use these to predict corresponding ``y`` values.
        However, the emulator can also be used to solve an inverse problem, where the user has an observation of ``y`` and wants to find the corresponding ``X``.
        Problems of this type are common in engineering and science, where the user has an observation of a system and wants to find the parameters that generated that observation.
        This operation can be numerically intensive, and the emulator can be used to solve this problem quickly and efficiently.

        Args:
            df_obs (pandas.DataFrame): A dataframe containing the single observation.
            df_std (pandas.DataFrame): A dataframe containing the error on the single observation.
            params (CalibrateParams, optional): A parameter configuration that contains all optional calibration parameters.
            wait (bool, optional): If ``True`` wait for the job to complete, otherwise return the process ID and exit.
            verbose (bool, optional): Display detailed information about the operation while running.

        Returns:
            pandas.DataFrame, str: By default, the solution to the inverse problem is either presented as a summary,
            or as the full set of points sampled from the posterior distribution.
            See the documentation for ``CalibrateParams`` for more information on the different options.
            Instead, if ``wait=False``, the process ID is returned.
            The results can then be retrieved later using ``Emulator.get_process(<process_id>)``.
            Process IDs associated with an emulator can be found using ``Emulator.list_processes()``.

        Example:
            .. code-block:: python

                emulator = tl.Emulator("quickstart")
                df_obs = pd.DataFrame({'y': [0.1]})
                df_std = pd.DataFrame({'y': [0.01]})
                emulator.calibrate(df_obs, df_std)

            .. code-block:: console

                    mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
                x  0.496  0.013   0.471    0.521        0.0      0.0    2025.0    2538.0    1.0

        """
        API_METHOD = "solve_inverse"
        if SYNC:
            csv = self._use_method(
                method=API_METHOD,
                df=df_obs,
                df_std=df_std,
                **params.unpack_parameters(),
                verbose=verbose,
            )
            df = pd.read_csv(csv, sep=",")
        else:
            data_csv = utils.get_csv_string(df_obs)
            data_std_csv = utils.get_csv_string(df_std)
            dataset_id = _upload_large_datasets(df_obs, data_csv, API_METHOD)
            dataset_std_id = _upload_large_datasets(
                df_std, data_std_csv, API_METHOD + "_std"
            )
            if dataset_id is not None:
                data_csv = None
            if dataset_std_id is not None:
                data_std_csv = None
            _, response = api.use_request_model(
                model_id=self.id,
                method=API_METHOD,
                data_csv=data_csv,
                data_std_csv=data_std_csv,
                dataset_id=dataset_id,
                dataset_std_id=dataset_std_id,
                **params.unpack_parameters(),
                processor=PROCESSOR,
                verbose=DEBUG,
            )
            process_id = utils.get_value_from_body("process_id", response)
            if verbose:
                print(f"Job {PROCESS_MAP[API_METHOD]} process ID: {process_id}")
            if not wait:
                return process_id
            _, response = _wait_for_job_completion(
                self.id, API_METHOD, process_id, verbose=verbose
            )
            csv = _retrieve_dataframe_from_response(response)
            df = _process_csv(csv, API_METHOD)
        if verbose:
            print("Calibration summary:")
            print(df)
        return df

    @typechecked
    def maximize(
        self,
        params: MaximizeParams = MaximizeParams(),
        wait: bool = True,
        verbose: bool = True,
    ) -> Union[pd.DataFrame, str]:
        """Finding the maximum the output of a trained emulator that exists on the twinLab cloud.

        This method of an emulator is used to find the point in the input space that maximizes the output of your trained emulator.
        This method is useful for finding the best possible input for your model, given the emulator predictions.
        This can help provide guidance for how best to configure your experiment, for example.
        This method can be used in a stand-alone manner to find the maximum, or at the end of a Bayesian optimization loop to find the best possible input.
        This method can also be used to minimize an output, by using the ``opt_weight`` argument of ``MaximizeParams`` to multiply the output by -1.
        If an emulator has more-than-one output, then a weighted combination of the outputs can be minimized/maximized.
        Once again, using the ``opt_weight`` argument of ``MaximizeParams`` can control the weight assigned to each output, or can be used to focus on a single output.

        Args:
            params (MaximizeParams): A parameter-configuration that contains optional parameters for finding the input that produces the maximum model output.
            wait (bool, optional): If ``True`` wait for the job to complete, otherwise return the process ID and exit.
            verbose (bool, optional): Display detailed information about the operation while running.

        Returns:
            Tuple[pandas.DataFrame], str: By default, a Dataframe containing the input that optimizes your emulator predictions.
            Instead, if ``wait=False``, the process ID is returned.
            The results can then be retrieved later using ``Emulator.get_process(<process_id>)``.
            Process IDs associated with an emulator can be found using ``Emulator.list_processes()``.

        Example:
            .. code-block:: python

                emulator = tl.Emulator("quickstart")
                emulator.maximize()

            .. code-block:: console

                          X
                0  0.845942

        """
        API_METHOD = "maximize"
        if SYNC:
            csv = self._use_method(
                method=API_METHOD,
                **params.unpack_parameters(),
                verbose=verbose,
            )
        else:
            _, response = api.use_request_model(
                model_id=self.id,
                method=API_METHOD,
                **params.unpack_parameters(),
                processor=PROCESSOR,
                verbose=DEBUG,
            )
            process_id = utils.get_value_from_body("process_id", response)
            if verbose:
                print(f"Job {PROCESS_MAP[API_METHOD]} process ID: {process_id}")
            if not wait:
                return process_id
            _, response = _wait_for_job_completion(
                self.id, API_METHOD, process_id, verbose=verbose
            )
            csv = utils.get_value_from_body("dataframe", response)
            csv = io.StringIO(csv)
        df = _process_csv(csv, API_METHOD)
        if verbose:
            print("Optimal input:")
            print(df)
        return df

    @typechecked
    def learn(
        self,
        dataset: Dataset,
        inputs: List[str],
        outputs: List[str],
        num_loops: int,
        num_points_per_loop: int,
        acq_func: str,
        simulation: Callable,  # A function that ingests X and returns y
        train_params: TrainParams = TrainParams(),
        recommend_params: RecommendParams = RecommendParams(),
        verbose: bool = True,
    ) -> None:
        """Perform active learning to improve an emulator on the twinLab cloud.

        Active learning is a method that can identify and utilise the most informative data points to add to an emulator in order to reduce the number of measurements to be taken or simulations that are required.
        Using active learning can result in a more accurate model, trained with less data.
        The primary difference between this method and ``Emulator.recommend`` is that in this method, the emulator is trained, new data points are suggested, and then training occurs continuously in an active loop.
        This way, new data can be used to train and update an emulator until the desired level of accuracy is achieved.
        This can be done using either the ``"optmise"`` or ``"explore"`` acquisition functions.
        The emulator is therefore updated on the twinLab cloud with the objective of either finding the point of maximum output or reducing the overall uncertainty in the emulator.
        This method does not return anything to the user directly, but instead updates the ``Dataset`` and ``Emulator`` in the cloud.

        Args:
            dataset (Dataset): twinLab dataset object which contains the initial training data for the emulator.
            inputs (list[str]): List of input column names in the training dataset.
            outputs (list[str]): List of output column names in the training dataset.
            num_loops (int): Number of loops to run of the learning process. This must be a positive integer. Note that in this method, the emulator is trained and then re-trained on new suggested data points, so setting ``num_loops=1`` here will mean that ``Emulator.train`` is run _twice_, and ``Emulator.recommend`` is run once.
            num_points_per_loop (int): Number of points to sample in each loop.
            acq_func (str): Specifies the acquisition function to be used when recommending new points: either ``"explore"`` or ``"optimise"``.
            simulation (Callable): A function that takes in a set of inputs and generates the outputs (for example, a simulator for the data generating process).
            train_params (TrainParams, optional): A parameter configuration that contains optional training parameters. Note that currently we only support the case when ``"test_train_ratio=1"`` when running a learning loop. Note that fixed-noise Gaussian Processes are not supported in this method and will raise an error. This includes: ``"fixed_noise_gp"``, ``"heteroskedastic_gp"``, ``"fixed_noise_multi_fidelity_gp"``.
            recommend_params (RecommendParams, optional): A parameter configuration that contains optional recommendation parameters.
            verbose (bool, optional): Display detailed information about the operation while running. If ``True``, the requested candidate points will be printed to the screen while running. If ``False`` the emulator will be updated on the cloud while the method runs silently.

        Examples:
            .. code-block:: python

                emulator = tl.Emulator("quickstart")
                dataset = tl.Dataset("quickstart")
                emulator.learn(
                    dataset=dataset,
                    inputs=["x"],
                    outputs=["y"],
                    num_loops=3,
                    num_points_per_loop=5,
                    acq_func="explore",
                    simulation=my_simulator,
                )

        """

        # Perform checks
        # NOTE: We could relax the train_test_ratio requirement in the future, but only easily with shuffle=False
        # and only by changing the train_test_ratio each iteration so that the test set is preserved
        if train_params.train_test_ratio != 1:
            raise ValueError(
                f"The test_train_ratio must be set to 1, not {train_params.train_test_ratio}, for this method to work."
            )
        if num_loops <= 0:
            raise ValueError(
                f"num_loops must be set to an integer value of 1 or more, not {num_loops}, for this method to work."
            )
        # NOTE: We could relax this in the future if we allowed for the simulation to output an error
        invalid_GP_estimators = [
            "fixed_noise_gp",
            "heteroskedastic_gp",
            "fixed_noise_multi_fidelity_gp",
        ]
        if train_params.estimator_params.estimator_type in invalid_GP_estimators:
            raise ValueError(
                f"The specified estimator type, {train_params.estimator_params.estimator_type}, is not currently supported. Please check the ``EstimatorParams`` documentation for more available estimator types."
            )

        # Train model initially
        self.train(
            dataset=dataset,
            inputs=inputs,
            outputs=outputs,
            params=train_params,
            wait=True,
            verbose=False,
        )

        # Loop over iterations of learning
        for i in range(num_loops):

            # Compute optimal sample location(s)
            candidate_points, acq_func_value = self.recommend(
                num_points=num_points_per_loop,
                acq_func=acq_func,
                params=recommend_params,
                wait=True,
                verbose=False,
            )

            # Evaluating the candidate points
            candidate_points[outputs] = simulation(candidate_points[inputs].values)

            # Download current training data, append new data, and reupload
            df_train = self.view_train_data()
            df_train = pd.concat([df_train, candidate_points], ignore_index=True)
            dataset.upload(df_train)

            # Train model
            self.train(
                dataset=dataset,
                inputs=inputs,
                outputs=outputs,
                params=train_params,
                wait=True,
                verbose=False,
            )

            # Write summary of the iteration if verbose
            if verbose:
                print(f"Iteration: {i+1}")
                print("Suggested candidate point(s):")
                pprint(candidate_points)
                print("Acquisition function value:")
                print(acq_func_value)
                print()

        # Print the estimated optimal point(s) after the final iteration if optimise and verbose
        if verbose and acq_func == "optimise":
            params = MaximizeParams(opt_weights=recommend_params.weights)
            df_max = self.maximize(params=params, verbose=False)
            print("Estimated optimal point(s):")
            print(df_max)
            print()

        # A nice final message if verbose
        if verbose:
            print(
                f"The candidate points have been uploaded in the Dataset {dataset.id}, alongside the emulator training data, on twinLab cloud."
            )

    @typechecked
    def delete(self, verbose: bool = False) -> None:
        """Delete emulator from the twinLab cloud.

        It can be useful to delete an emulator to keep a cloud account tidy, or if an emulator is no longer necessary.

        Args:
            verbose (bool, optional): Display detailed information about the operation while running.

        Examples:
            .. code-block:: python

                emulator = tl.Emulator("quickstart")
                emulator.delete()

        """
        _, response = api.delete_model(self.id, verbose=DEBUG)
        if verbose:
            message = utils.get_message(response)
            print(message)

    @typechecked
    def plot(
        self,
        x_axis: str,
        y_axis: str,
        x_fixed: Dict[str, float] = {},
        params: PredictParams = PredictParams(),
        x_lim: Optional[Tuple[float, float]] = None,
        n_points: int = 100,
        label: str = "Emulator",
        color: str = digilab_colors["light_blue"],
        verbose: bool = False,
    ) -> plt.plot:
        """Plot the predictions from an emulator across a single dimension with one and two standard deviation bands.

        This will make a call to the emulator to predict across the specified dimension.
        Note that a multi-dimensional emulator will be sliced across the other dimensions.
        The matplotlib.pyplot object is returned, and can be further modified by the user.

        Args:
            x_axis (str): The name of the x-axis variable.
            y_axis (str): The name of the y-axis variable.
            x_fixed (dict, optional): A dictionary of fixed values for the other X variables.
                Note that all X variables of an emulator must either be specified as x_axis or appear as x_fixed keys.
                To pass through "None". either leave x_fixed out or pass through an empty dictionary.
            params: (PredictParams, optional). A parameter configuration that contains optional prediction parameters.
            x_lim (tuple[float, float], optional]: The limits of the x-axis.
                If not provided. the limits will be taken directly from the emulator.
            n_points (int, optional): The number of points to sample in the x-axis.
            label (str, optional): The label for the line in the plot. defaults to "Emulator prediction".
            color (str, optional): The color of the plot. Defaults to digiLab blue.
                Can be any valid matplotlib color (https://matplotlib.org/stable/gallery/color/named_colors.html).
            verbose (bool, optional): Display detailed information about the operation while running.

        Examples:
            .. code-block:: python

                emulator = tl.Emulator("emulator_id")
                plt = emulator.plot("Time", "Temperature", x_fixed={"Latitude": 0, "Longitude": 30})
                plt.show()

        """

        # Get information about inputs/outputs from the emulator
        _, response = api.summarise_model(self.id, verbose=DEBUG)
        inputs = set(response["model_summary"]["data_diagnostics"]["inputs"].keys())
        outputs = set(response["model_summary"]["data_diagnostics"]["outputs"].keys())

        # Check function inputs
        if x_axis not in inputs:
            raise ValueError(f"x_axis must be one of the Emulator inputs: {inputs}")
        if y_axis not in outputs:
            raise ValueError(f"y_axis must be one of the Emulator outputs: {outputs}")
        if set([x_axis] + list(x_fixed.keys())) != inputs:
            raise ValueError(
                f"All values {inputs} must be specified as either x_axis or x_fixed keys"
            )

        # Get the range for the x-axis
        if x_lim is not None:
            xmin, xmax = x_lim
        else:
            inputs = response["model_summary"]["data_diagnostics"]["inputs"]
            xmin, xmax = inputs[x_axis]["min"], inputs[x_axis]["max"]

        # Create a dataframe on which to predict
        X = {x_axis: np.linspace(xmin, xmax, n_points)}
        for x_col, x_val in x_fixed.items():
            X[x_col] = x_val * np.ones(n_points)
        df_X = pd.DataFrame(X)

        # Predict using the emulator
        df_mean, df_std = self.predict(
            df_X,
            params=params,
            verbose=verbose,
        )

        # Plot the results
        plt = plot(x_axis, y_axis, df_X, df_mean, df_std, color=color, label=label)
        return plt  # Return the plot

    @typechecked
    def heatmap(
        self,
        x1_axis: str,
        x2_axis: str,
        y_axis: str,
        x_fixed: Dict[str, float] = {},
        mean_or_std: str = "mean",
        params: PredictParams = PredictParams(),
        x1_lim: Optional[Tuple[float, float]] = None,
        x2_lim: Optional[Tuple[float, float]] = None,
        n_points: int = 25,
        cmap=digilab_cmap,
        verbose: bool = False,
    ) -> plt.plot:
        """Plot a heatmap of the predictions from an emulator across two dimensions.

        This will make a call to the emulator to predict across the specified dimensions.
        Note that a higher-than-two-dimensional emulator will be sliced across the other dimensions.
        The matplotlib.pyplot object is returned, and can be further modified by the user.
        The uncertainty of the emulator is not plotted here.

        Args:
            x1_axis (str): The name of the x1-axis variable (horizonal axis).
            x2_axis (str): The name of the x2-axis variable (vertical axis).
            y_axis (str): The name of the plotted variable (heatmap).
            x_fixed (dict, optional): A dictionary of fixed values for the other ``X`` variables.
                Note that all ``X`` variables of an emulator must either be specified as ``x1_axis``, ``x2_axis`` or appear as keys in ``x_fixed``.
                Passing an empty dictionary (the default) will fix none of the variables.
            mean_or_std (str, optional): A string determining whether to plot the mean (``"mean"``) or standard deviation (``"std"``) of the emulator. Defaults to ``"mean"``.
            params: (PredictParams, optional). A parameter configuration that contains optional prediction parameters.
            x1_lim (tuple[float, float], optional): The limits of the x1-axis.
                If not provided, the limits will be taken directly from the emulator.
            x2_lim (tuple[float, float], optional): The limits of the x2-axis.
                If not provided, the limits will be taken directly from the emulator.
            n_points (int, optional): The number of points to sample in each dimension.
                The default is 25, which will create a 25x25 grid.
            cmap (str, optional): The color of the plot. Defaults to a digiLab palette.
                Can be any valid matplotlib color (https://matplotlib.org/stable/users/explain/colors/colormaps.html).
            verbose (bool, optional): Display detailed information about the operation while running.

        Returns:
            matplotlib.pyplot: Matplotlib plot object

        Examples:
            .. code-block:: python

                emulator = tl.Emulator("emulator_id") # A trained emulator
                plt = emulator.heatmap("Latitude", "Longitude", "Rainfall", x_fixed={"Month": 6})
                plt.show()

        """

        # Get information about inputs/outputs from the emulator
        _, response = api.summarise_model(self.id, verbose=DEBUG)
        inputs = set(response["model_summary"]["data_diagnostics"]["inputs"].keys())
        outputs = set(response["model_summary"]["data_diagnostics"]["outputs"].keys())

        # Check function inputs
        if x1_axis not in inputs:
            raise ValueError(f"x1_axis must be one of the Emulator inputs:{inputs}")
        if x2_axis not in inputs:
            raise ValueError(f"x2_axis must be one of the Emulator inputs: {inputs}")
        if y_axis not in outputs:
            raise ValueError(f"y_axis must be one of the Emulator outputs: {outputs}")
        if set([x1_axis, x2_axis] + list(x_fixed.keys())) != inputs:
            raise ValueError(
                f"All values {inputs} must be specified as either x1_axis, x2_axis, or x_fixed keys"
            )

        # Get the ranges for the x-axes
        inputs = response["model_summary"]["data_diagnostics"]["inputs"]
        if x1_lim is None:
            x1min, x1max = inputs[x1_axis]["min"], inputs[x1_axis]["max"]
        else:
            x1min, x1max = x1_lim
        if x2_lim is None:
            x2min, x2max = inputs[x2_axis]["min"], inputs[x2_axis]["max"]
        else:
            x2min, x2max = x2_lim

        # Create a grid of points
        x1 = np.linspace(x1min, x1max, n_points)
        x2 = np.linspace(x2min, x2max, n_points)
        X1, X2 = np.meshgrid(x1, x2)

        # Create a dataframe on which to predict
        X = {x1_axis: X1.flatten(), x2_axis: X2.flatten()}
        if x_fixed is not None:
            for x_col, x_val in x_fixed.items():
                X[x_col] = x_val * np.ones(n_points**2)
        df_X = pd.DataFrame(X)

        # Predict using the emulator
        # NOTE: Uncertainty is discarded here
        df_mean, df_std = self.predict(df_X, params=params, verbose=verbose)

        if mean_or_std == "mean":
            df = df_mean
        elif mean_or_std == "std":
            df = df_std
        else:
            raise ValueError("mean_or_std must be either 'mean' or 'std'")

        # Plot the results
        plt = heatmap(
            x1_axis,
            x2_axis,
            y_axis,
            df_X,
            df,
            cmap,
        )
        return plt  # Return the plot
