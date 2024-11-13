# Standard imports
import io
import os
import sys
import time
import uuid
from pprint import pprint
from typing import Callable, Dict, List, Optional, Tuple, Union

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typeguard import typechecked

# Project imports
from . import _api, _plotting, _utils, settings
from ._beta.params import TrainParamsBeta
from ._plotting import DIGILAB_CMAP as digilab_cmap
from ._plotting import DIGILAB_COLORS as digilab_colors
from ._utils import (
    EmulatorResultsAdapter,
    convert_time_formats_in_status,
    download_file_from_url,
    match_project,
)
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
from .prior import Prior
from .settings import ValidExportFormats, ValidStatus, ValidFMUOS, ValidFMUTypes

# Parameters
ACQ_FUNC_DICT = {  # TODO: Delete this?
    "ExpectedImprovement": "EI",
    "qExpectedImprovement": "qEI",
    "LogExpectedImprovement": "LogEI",
    "qLogExpectedImprovement": "qLogEI",
    "PosteriorStandardDeviation": "PSD",
    "qPosteriorStandardDeviation": "qPSD",
    "qNegIntegratedPosteriorVariance": "qNIPV",
}
PING_TIME_INITIAL = 1.0  # Seconds
PING_FRACTIONAL_INCREASE = 0.1
PROCESSOR = "cpu"
PROCESS_MAP = {
    "score": "score",
    "get_calibration_curve": "benchmark",
    "predict": "predict",
    "sample": "sample",
    "get_candidate_points": "recommend",
    "solve_inverse": "calibrate",
    "maximize": "maximize",
}
NOT_WAIT_TIME = 5  # Seconds; check a job has dispatched okay
ALLOWED_DATAFRAME_SIZE = 5.5 * int(1e6)  # Safely withing 6MB AWS limit

### Helper functions ###


def _dataset_over_memory_limit(dataset_csv: str) -> pd.DataFrame:
    if sys.getsizeof(dataset_csv) > ALLOWED_DATAFRAME_SIZE:
        return True
    else:
        return False


def _process_request_dataframes(
    df: pd.DataFrame, params: dict, dataset_name: str, project_id: str
) -> dict:

    # Create a unique ID for the dataset
    dataset_id = str(uuid.uuid4())

    # Convert the dataframe to a CSV string
    dataset_csv_string = _utils.get_csv_string(df)

    # If the dataset is too large, upload it using a URL and pass the dataset ID in the request parameters
    if _dataset_over_memory_limit(dataset_csv_string):

        # Generate a temporary upload URL - this URL does not have the 5.5mb limit
        _, response = _api.get_dataset_temporary_upload_url(project_id, dataset_id)
        url = _utils.get_value_from_body("url", response)

        # Upload the dataframe to the presigned URL
        _utils.upload_dataframe_to_presigned_url(df, url, check=settings.CHECK_DATASETS)

        params[f"{dataset_name}_id"] = dataset_id

    else:

        # Pass the dataset as a CSV string in the request parameters
        params[dataset_name] = dataset_csv_string

    return params


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
        project (str): Name of the project to which the emulator belongs. Defaults to "personal".
        project_owner (str): Email address of the project owner. Defaults to the current user.
    """

    @typechecked
    def __init__(
        self, id: str, project: str = "personal", project_owner: Optional[str] = None
    ):
        self.id = id
        self.project_id = match_project(project, project_owner)

    # TODO: DO designs
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
        _, request_response = _api.post_design(
            serialised_priors,
            params.sampling_method.to_json(),
            num_points,
            seed=params.seed,
        )

        # Get the process_id from the response body
        process_id = _utils.get_value_from_body("process_id", request_response)

        # Wait for the process to complete
        response = _utils.wait_for_job_completion(
            _api.get_design, process_id, verbose=verbose
        )

        initial_design_df = EmulatorResultsAdapter("design", response).adapt_result(
            verbose=verbose
        )

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
        params: Union[TrainParams, TrainParamsBeta] = TrainParams(),
        wait: bool = True,
        verbose: bool = True,
    ):
        """Train an emulator on the twinLab cloud.

        This is the primary functionality of twinLab, whereby an emulator is trained to learn patterns from a dataset.
        The emulator learns trends in the dataset and then is able to make predictions on new data.
        These new data may be far away from the training data;
        the emulator will effectively interpolate between the training data points.
        The emulator can also be used to extrapolate beyond the training data, but this is less reliable.

        Emulators can be trained on datasets with multiple inputs and outputs,
        and can be used to make predictions on new data with multiple inputs and outputs.
        The powerful algorithms in twinLab allow for the emulator to not only make predictions,
        but to also quantify the uncertainty in these predictions.
        This is extremely advantageous, because it allows for the reliability of the predictions to be quantified.

        The training process will start on the twinLab cloud using the ID of the emulator previously instantiated (with `tl.Emulator(id=)`).
        If that emulator has not been trained already, the training process will start directly. Otherwise, the process will required you to rename the emulator or delete the existing one (with `Emulator.delete()`).

        See the documentation for :func:`~params.TrainParams` for more information on the available training parameters.
        The documentation for :func:`~params.EstimatorParams` contains information about estimator types and kernels.
        Finally, the documentation for :func:`~params.ModelSelectionParams` details automatic model selection parameters.

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

            Train a simple emulator:

            .. code-block:: python

                df = pd.DataFrame({"X": [1, 2, 3, 4], "y": [1, 4, 9, 16]})
                dataset = tl.Dataset("my_dataset")
                dataset.upload(df)
                emulator = tl.Emulator("my_emulator")
                emulator.train(dataset, ["X"], ["y"])

            Train a emulator with dimensionality reduction (here on the output):

            .. code-block:: python

                dataset = tl.Dataset("my_dataset")
                emulator = tl.Emulator("my_emulator")
                params = tl.TrainParams(output_retained_dimensions=1)
                emulator.train(dataset, ["X"], ["y1", "y2"], params)

            Train an emulator with a specified (here variational) estimator type:

            .. code-block:: python

                dataset = tl.Dataset("my_dataset")
                emulator = tl.Emulator("my_emulator")
                estimator_params=tl.EstimatorParams(estimator_type="variational_gp")
                params = tl.TrainParams(estimator_params=estimator_params)
                emulator.train(dataset, ["X"], ["y"], params)

            Train an emulator with a specific (here linear) kernel:

            .. code-block:: python

                dataset = tl.Dataset("my_dataset")
                emulator = tl.Emulator("my_emulator")
                params = tl.TrainParams(estimator_params=tl.EstimatorParams(kernel="LIN"))
                emulator.train(dataset, ["X"], ["y"], params)

            Train an emulator using automatic kernel selection to find the best kernel:

            .. code-block:: python

                dataset = tl.Dataset("my_dataset")
                emulator = tl.Emulator("my_emulator")
                params = tl.TrainParams(model_selection=True)
                emulator.train(dataset, ["X"], ["y"], params)

        """

        # Making a dictionary from TrainParams class
        if PROCESSOR == "gpu":
            print(
                "Emulator is being trained on GPU. Inference operations must also be performed on GPU"
            )
        emulator_params, training_params = params.unpack_parameters()
        emulator_params["inputs"] = inputs
        emulator_params["outputs"] = outputs
        training_params["dataset_id"] = dataset.id

        # Send training request
        _, request_response = _api.post_emulator(
            self.project_id,
            self.id,
            emulator_params,
            training_params,
            processor=PROCESSOR,
        )
        if verbose:
            detail = _utils.get_value_from_body("detail", request_response)
            print(detail)
        if not wait:
            time.sleep(NOT_WAIT_TIME)
            _api.get_emulator_status(self.project_id, self.id)
        else:
            _utils.wait_for_job_completion(
                _api.get_emulator_status, self.project_id, self.id, verbose=verbose
            )
            if verbose:
                print(f"Training of emulator {self.id} is complete!")

    @typechecked
    def status(self, verbose: bool = False) -> dict:
        """Check the status of a training process on the twinLab cloud.

        Args:
            verbose (bool, optional): Display information about the operation while running.

        Returns:
            Tuple[int, dict]: A tuple containing the status code and the response body.

        Example:
            .. code-block:: python

                emulator = tl.Emulator("my_emulator")
                emulator.status()

            .. code-block:: console

                {
                    'status': 'success'
                    'start_time': '2024-07-31 18:12:33',
                    'end_time': '2024-07-31 18:12:35',
                }

        """
        _, response = _api.get_emulator_status(self.project_id, self.id)
        status_dict = _utils.get_value_from_body("status_detail", response)

        # Convert the time stamps into a nicer format
        if status_dict.get("start_time"):
            status_dict["start_time"] = _utils.convert_time_format(
                status_dict["start_time"]
            )
        if status_dict.get("end_time"):
            status_dict["end_time"] = _utils.convert_time_format(
                status_dict["end_time"]
            )

        if verbose:
            detail = _utils.get_value_from_body("detail", response)
            print(detail)
        return status_dict

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

        _, response = _api.get_emulator_parameters(self.project_id, self.id)
        parameters = _utils.get_value_from_body("parameters", response)
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
        _, response = _api.get_emulator_data(
            self.project_id, self.id, dataset_type="training_data"
        )

        if response.get("dataset"):
            csv = _utils.get_value_from_body("dataset", response)
            csv = io.StringIO(csv)
            df_train = pd.read_csv(csv, sep=",", index_col=0)
        elif response["dataset_url"]:
            url = response["dataset_url"]
            data_json = _utils.download_dataframe_from_presigned_url(url)
            csv = data_json["training_data"]
            df_train = pd.read_csv(csv, sep=",", index_col=0)
        else:
            raise ValueError("No dataset found in the response.")
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
        _, response = _api.get_emulator_data(
            self.project_id, self.id, dataset_type="testing_data"
        )

        if response.get("dataset"):
            csv = _utils.get_value_from_body("dataset", response)
            csv = io.StringIO(csv)
            df_test = pd.read_csv(csv, sep=",", index_col=0)
        elif response["dataset_url"]:
            url = response["dataset_url"]
            data_json = _utils.download_dataframe_from_presigned_url(url)
            csv = data_json["testing_data"]
            df_test = pd.read_csv(csv, sep=",", index_col=0)
        if verbose:
            print("Test data")
            pprint(df_test)
        return df_test

    def list_processes(self, verbose: bool = False) -> List[str]:
        """List all of the processes associated with a given emulator on the twinLab cloud.

        Args:
            verbose (bool, optional): Determining level of information returned to the user. Default is False.

        Returns:
            list: List containing all of the process IDs associated with the emulator.

        Example:

            .. code-block:: python

                emulator = tl.Emulator("quickstart")
                emulator.list_processes()

            .. code-block:: console

                ['predict-distinct-honey-milk', 'sample-four-hungry-wolves']

        """
        _, response = _api.get_emulator_processes(self.project_id, self.id)
        processes = _utils.get_value_from_body("process_ids", response)

        if verbose:
            print(processes)

        return processes

    def list_processes_statuses(self, verbose: bool = False) -> List[str]:
        """List the status of all processes associated with a given emulator on the twinLab cloud.

        This includes the current status of the process, the start time, the end time, and the process ID.

        Args:
            verbose (bool, optional): Determining level of information returned to the user. Default is False.

        Returns:
            dict: Dictionary containing all processes associated with the emulator.

        Example:

            .. code-block:: python

                emulator = tl.Emulator("quickstart")
                emulator.list_processes_statuses()

            .. code-block:: console

                [
                    {
                        'method': 'sample',
                        'process_id': 'sample-four-hungry-wolves',
                        'run_time': '0:00:05',
                        'start_time': '2024-04-09 17:10:12',
                        'status': 'success'
                    },
                    {
                        'method': 'predict',
                        'process_id': 'predict-distinct-honey-milk',
                        'run_time': '0:00:04',
                        'start_time': '2024-04-09 18:45:48',
                        'status': 'success'
                    },
                ]

        """
        _, response = _api.get_emulator_processes_statuses(self.project_id, self.id)
        process_statuses = _utils.get_value_from_body("process_statuses", response)

        # Print detailed emulator information to screen
        if verbose:

            # Create dictionary of cuddly status messages
            status_messages = {
                ValidStatus.PROCESSING.value: "Processes currently runnning:",
                ValidStatus.SUCCESS.value: "Successful processes:",
                ValidStatus.FAILURE.value: "Processes that failed to complete:",
                None: "Processes with unknown status:",
            }
            if process_statuses:
                for nice_status in status_messages.keys():
                    print("\033[1m" + status_messages[nice_status])  # Bold text
                    print("\033[0m")  # Reset text formatting
                    status_count = 0
                    for status_dict in process_statuses:
                        status = status_dict.get("status", None)
                        if status == nice_status:
                            status_count += 1
                            if status_dict.get("end_time", None):
                                status_dict["run_time"] = _utils.calculate_runtime(
                                    status_dict.get("start_time"),
                                    status_dict.pop("end_time"),
                                )
                            else:
                                status_dict["run_time"] = "N/A"
                            status_dict = convert_time_formats_in_status(status_dict)
                            pprint(status_dict, compact=True, sort_dicts=False)
                            print()
                    if status_count == 0:
                        print(f"No {status_messages[nice_status].lower()[:-1]}")
                        print()
            else:
                print("No processes available for this emulator.")

        return process_statuses

    @typechecked
    def get_process(self, process_id: str, verbose: bool = False) -> Union[
        None,  # score; benchmark with no test data
        float,  # score (with combined_score=True)
        pd.DataFrame,  # score; benchmark; sample; calibrate; maximize
        Tuple[pd.DataFrame, pd.DataFrame],  # predict
        Tuple[pd.DataFrame, float],  # recommend
        str,  # message for failed/running process
    ]:
        """Get the results from a process associated with the emulator on the twinLab cloud.

        This allows a user to retrieve any results from processes (jobs) they have run previously.
        The list of available process IDs can be obtained from the ``list_processes()`` method.

        Args:
            process_id (str): The ID of the process from which to get the results.
            verbose (bool, optional): Display information about the operation while running.

        Example:
            .. code-block:: python

                emulator = tl.Emulator("quickstart")
                emulator.get_process("sample-four-hungry-wolves")

            .. code-block:: console

                            y
                            0         1         2         3
                0   -0.730114  0.474193  0.046743  1.327620
                1   -0.656061  0.505923  0.074198  1.289113
                2   -0.579500  0.538610  0.100665  1.247405
                3   -0.502726  0.574996  0.128068  1.205057
                4   -0.428691  0.614687  0.157740  1.165903

        """
        # Get the status of the process
        _, response = _api.get_emulator_process(self.project_id, self.id, process_id)
        status = ValidStatus(response["status"])

        if status is ValidStatus.FAILURE:
            return f"Process {process_id} has failed to complete"
        elif status is ValidStatus.PROCESSING:
            return f"Process {process_id} is still running"
        else:

            # Get the result of the process
            _, response = _api.get_emulator_process(
                self.project_id, self.id, process_id
            )

            # Process the response according to the method used
            method = process_id.split("-")[0]
            result = EmulatorResultsAdapter(method, response).adapt_result(
                verbose=verbose
            )
            if verbose:
                print(f"Process {process_id} results:")
                print(result)
            return result

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
        _, response = _api.get_emulator_summary(self.project_id, self.id)
        summary = _utils.get_value_from_body("summary", response)
        del summary["data_diagnostics"]

        if "base_estimator_diagnostics" in summary["estimator_diagnostics"].keys():
            base_estimator_summary = _utils.reformat_summary_dict(
                summary, detailed=detailed
            )
            summary["estimator_diagnostics"][
                "base_estimator_diagnostics"
            ] = base_estimator_summary

        # If the emulator is a mixture of experts, access each expert, reformat and put them in a dictionary

        elif "gp_classifier" in summary["estimator_diagnostics"].keys():
            all_summaries = []
            for key, value in summary["estimator_diagnostics"].items():
                if key.startswith("gp_"):
                    gp = {}
                    gp["estimator_diagnostics"] = value
                    summary = _utils.reformat_summary_dict(gp, detailed=detailed)
                    summary = {key: summary}
                    all_summaries.append(summary)

            gp_experts_dictionary = {}
            for summary in all_summaries:
                gp_experts_dictionary.update(summary)
                summary = gp_experts_dictionary

        else:
            summary = _utils.reformat_summary_dict(summary, detailed=detailed)
        if verbose:
            print("Trained emulator summary:")
            pprint(summary, compact=True, sort_dicts=False)
        return summary

    @typechecked
    def update(
        self,
        df: pd.DataFrame,
        df_std: Optional[pd.DataFrame] = None,
        wait: bool = True,
        verbose: bool = False,
    ) -> Union[None, str]:
        """
        Update an emulator with new training data.

        This method allows a user to update an existing emulator with new training data.
        This is useful when new data is collected, and the emulator needs to incorporate this new information, but you do not want to train the emulator from scratch.
        The parameters of the emulator are not updated; in this process the emulator is simply conditioned on the new data points.
        This means that the process is fast, but note that the quality of the emulator may degrade with multiple frequent updates, compared to re-training in an active learning loop.

        Args:
            df (pandas.DataFrame): The new training data to update the emulator with.
            df_std (pandas.DataFrame, optional): The standard deviation of the new training data.
                This is used to update the emulator with uncertainty information.
            wait (bool, optional): If ``True`` wait for the job to complete, otherwise return the process ID and exit.
            verbose (bool, optional): Display information about the operation while running.

        Returns:
            None

        Example:

            .. code-block:: python

                df = pd.DataFrame({"X": [1, 2, 3, 4], "y": [1, 4, 9, 16]})
                emulator = tl.Emulator("my_emulator")
                emulator.train(df)

                # Update the emulator with new data
                new_df = pd.DataFrame({"X": [5], "y": [25]})
                emulator.update(new_df)
        """

        params = {}
        params = _process_request_dataframes(df, params, "dataset", self.project_id)
        if df_std is not None:
            params = _process_request_dataframes(
                df_std, params, "dataset_std", self.project_id
            )

        _, body = _api.post_emulator_update(self.project_id, self.id, params)

        process_id = body["process_id"]
        if verbose:
            print(f"Job update process ID: {process_id}")

        if not wait:
            time.sleep(NOT_WAIT_TIME)
            _api.get_emulator_process(self.project_id, self.id, process_id)
            return process_id

        # Await the completion of the scoring process
        response = _utils.wait_for_job_completion(
            _api.get_emulator_process,
            self.project_id,
            self.id,
            process_id,
            verbose=verbose,
        )

        update = EmulatorResultsAdapter("update", response).adapt_result(
            verbose=verbose
        )

        if update:
            print(f"Your emulator has been updated with new data.")

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
        The score can be calculated using different metrics; see the :func:`~params.ScoreParams` class for a full list and description of available metrics.

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

        _, response = _api.post_emulator_score(
            self.project_id, self.id, params.unpack_parameters()
        )

        process_id = _utils.get_value_from_body("process_id", response)

        # Await the completion of the scoring process
        response = _utils.wait_for_job_completion(
            _api.get_emulator_process,
            self.project_id,
            self.id,
            process_id,
            verbose=verbose,
        )

        score = EmulatorResultsAdapter("score", response).adapt_result(verbose=verbose)

        return score

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
        See the documentation for :func:`~params.BenchmarkParams` for more information on the available benchmark types.

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
        _, response = _api.post_emulator_benchmark(
            self.project_id, self.id, params.unpack_parameters()
        )

        process_id = _utils.get_value_from_body("process_id", response)

        # Await the completion of the process
        response = _utils.wait_for_job_completion(
            _api.get_emulator_process,
            self.project_id,
            self.id,
            process_id,
            verbose=verbose,
        )

        benchmark = EmulatorResultsAdapter("benchmark", response).adapt_result(
            verbose=verbose
        )

        return benchmark

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
        The emulator returns both a predicted mean and an uncertainty for each output dimension.
        This allows a user to not only make predictions, but also to quantify the uncertainty on those predictions:

        - For a regression model the uncertainty is the Gaussian standard deviation, which is a measure of the uncertainty in the prediction.
          The emulator is 95% confident that the true value lies within two standard deviations of the mean.
        - For a classification model the uncertainty is the probability of the predicted class, which is a measure of the confidence in the prediction.

        If you are using an emulator trained on data from sources with different fidelity levels (a multi-fidelity emulator), the predictions will assume the highest fidelity level (fidelity = 1) for the output(s).

        See the documentation for :func:`~params.PredictParams` for more information on additional parameters that can be passed to predict.

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

        unpacked_params = _process_request_dataframes(
            df, params.unpack_parameters(), "dataset", self.project_id
        )
        _, body = _api.post_emulator_predict(self.project_id, self.id, unpacked_params)

        process_id = body["process_id"]
        if verbose:
            print(f"Job predict process ID: {process_id}")

        if not wait:
            time.sleep(NOT_WAIT_TIME)
            _api.get_emulator_process(self.project_id, self.id, process_id)
            return process_id

        response = _utils.wait_for_job_completion(
            _api.get_emulator_process,
            self.project_id,
            self.id,
            process_id,
            verbose=verbose,
        )

        df_mean, df_std = EmulatorResultsAdapter("predict", response).adapt_result(
            verbose=verbose
        )

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

        A secondary functionality of the emulator is to draw a set of example predictions from the trained emulator.
        Rather than quantifying the uncertainty in the predictions, this method draws representative samples from the emulator.
        The collection of samples can be used to explore the distribution of the emulator predictions.
        Each sample is a possible prediction of the emulator, and therefore a prediction of a possible new observation from the data-generation process.
        The covariance in the emulator predictions can therefore be explored, which is particularly useful for functional emulators.
        See the documentation for :func:`~params.SampleParams` for more information on additional parameters.

        If you are using an emulator trained on data from sources with different fidelity levels (a multi-fidelity emulator), the sample(s) will assume the highest fidelity level (fidelity = 1) for the output(s).

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
        unpacked_params = _process_request_dataframes(
            df, params.unpack_parameters(), "dataset", self.project_id
        )
        unpacked_params["num_samples"] = num_samples

        _, response = _api.post_emulator_sample(
            self.project_id, self.id, unpacked_params
        )

        process_id = _utils.get_value_from_body("process_id", response)
        if verbose:
            print(f"Job sample process ID: {process_id}")

        if not wait:
            time.sleep(NOT_WAIT_TIME)
            _api.get_emulator_process(self.project_id, self.id, process_id)
            return process_id
        else:
            response = _utils.wait_for_job_completion(
                _api.get_emulator_process,
                self.project_id,
                self.id,
                process_id,
                verbose=verbose,
            )
            df = EmulatorResultsAdapter("sample", response).adapt_result(
                verbose=verbose
            )
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
          This method can also be used to minimize an output, by using the ``weights`` argument of ``RecommendParams`` to multiply the output by -1.
          If an emulator has multiple outputs, then a weighted combination of the outputs can be minimized/maximized.
          The ``weights`` argument of ``RecommendParams`` can control the weight assigned to each output, or can be used to focus on a single output.
          For example, this can be used to find the maximum strength of a pipe given a set of design parameters.
        - ``"explore"`` will instead suggest ``"X"`` that reduce the overall uncertainty of the emulator across the entire input space.
          A classic use case for this would be a user trying to reduce overally uncertainty.
          For example, a user trying to reduce the uncertainty in the strength of a pipe across all design parameters.

        The number of requested data points can be specified by the user, and if this is greater than 1 then recommendations are all suggested at simultaneously, and are designed to be the optimal set, as a group, to achieve the user outcome.
        twinLab optimises which specific acquisition function within the chosen category will be used, prioritising numerical stability based on the number of points requested.
        See the documentation for :func:`~params.RecommendParams` for more information on the available parameters.

        For the ``"explore"`` functionality, generating recommendations is not supported for multi-output emulators (when ``"y"`` has more than one dimension).

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

        # Convert acq_func names to correct method depending on number of points requested
        if acq_func == "optimise":
            if num_points == 1:
                acq_func = "LogEI"
            else:
                acq_func = "qLogEI"
        if acq_func == "explore":
            if num_points == 1:
                acq_func = "PSD"
            else:
                acq_func = "qPSD"

        unpacked_params = params.unpack_parameters()
        unpacked_params["num_points"] = num_points
        if acq_func in ACQ_FUNC_DICT:
            unpacked_params["acq_func"] = ACQ_FUNC_DICT[acq_func]
        else:
            unpacked_params["acq_func"] = acq_func

        _, response = _api.post_emulator_recommend(
            self.project_id, self.id, unpacked_params
        )

        process_id = _utils.get_value_from_body("process_id", response)
        if verbose:
            print(f"Job recommend process ID: {process_id}")

        if not wait:
            time.sleep(NOT_WAIT_TIME)
            _api.get_emulator_process(self.project_id, self.id, process_id)
            return process_id
        else:
            response = _utils.wait_for_job_completion(
                _api.get_emulator_process,
                self.project_id,
                self.id,
                process_id,
                verbose=verbose,
            )
            df, acq_func_value = EmulatorResultsAdapter(
                "recommend", response
            ).adapt_result(verbose=verbose)
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
        See the documentation for :func:`~params.CalibrateParams` for more information on the available parameters.

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
        # Enforce limit on number of chains.
        # Prevents the user from starting excessively long jobs.
        if params.n_chains < 1 or params.n_chains > 4:
            raise ValueError("Number of chains must be between 1 and 4.")

        unpacked_params = _process_request_dataframes(
            df_obs, params.unpack_parameters(), "dataset_obs", self.project_id
        )
        unpacked_params = _process_request_dataframes(
            df_std, unpacked_params, "dataset_obs_std", self.project_id
        )

        _, response = _api.post_emulator_calibrate(
            self.project_id, self.id, unpacked_params
        )

        process_id = _utils.get_value_from_body("process_id", response)
        if verbose:
            print(f"Job calibrate process ID: {process_id}")

        if not wait:
            time.sleep(NOT_WAIT_TIME)
            _api.get_emulator_process(self.project_id, self.id, process_id)
            return process_id
        else:
            response = _utils.wait_for_job_completion(
                _api.get_emulator_process,
                self.project_id,
                self.id,
                process_id,
                verbose=verbose,
            )
            df = EmulatorResultsAdapter("calibrate", response).adapt_result(
                verbose=verbose
            )
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
        See the documentation for :func:`~params.MaximizeParams` for more information on the available parameters.

        This method can also be used to minimize an output, by using the ``opt_weight`` argument of ``MaximizeParams`` to multiply the output by -1.
        If an emulator has more-than-one output, then a weighted combination of the outputs can be minimized/maximized:
        The ``opt_weight`` argument of ``MaximizeParams`` can control the weight assigned to each output, or can be used to focus on a single output.

        Args:
            params (MaximizeParams): A parameter-configuration that contains optional parameters for finding the input that produces the maximum model output.
            wait (bool, optional): If ``True`` wait for the job to complete, otherwise return the process ID and exit.
            verbose (bool, optional): Display detailed information about the operation while running.

        Returns:
            Tuple[pandas.DataFrame], str: By default, a Dataframe containing the input that optimises your emulator predictions.
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
        _, response = _api.post_emulator_maximize(
            self.project_id, self.id, params.unpack_parameters()
        )

        process_id = _utils.get_value_from_body("process_id", response)
        if verbose:
            print(f"Job maximize process ID: {process_id}")

        if not wait:
            time.sleep(NOT_WAIT_TIME)
            _api.get_emulator_process(self.project_id, self.id, process_id)
            return process_id
        else:
            response = _utils.wait_for_job_completion(
                _api.get_emulator_process,
                self.project_id,
                self.id,
                process_id,
                verbose=verbose,
            )
            df = EmulatorResultsAdapter("maximize", response).adapt_result(
                verbose=verbose
            )
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
            train_params (TrainParams, optional): A parameter configuration that contains optional training parameters. Note that currently we only support the case when ``"test_train_ratio=1"`` when running a learning loop. Note that fixed-noise emulators are not supported in this method and will raise an error. This includes: ``"fixed_noise_gp"``, ``"heteroskedastic_gp"``, ``"fixed_noise_multi_fidelity_gp"``.
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

            # Append new data to the dataset
            dataset.append(candidate_points)
            # Delete the previous emulator before training again with the extra data. This is neccesary to use the same id.
            self.delete()
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
                pprint(candidate_points[inputs])
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
        _, response = _api.delete_emulator(self.project_id, self.id)
        if verbose:
            detail = _utils.get_value_from_body("detail", response)
            print(detail)

    @typechecked
    def plot(
        self,
        x_axis: str,
        y_axis: str,
        x_fixed: Dict[str, float] = {},
        params: PredictParams = PredictParams(),
        x_lim: Tuple[Optional[float], Optional[float]] = (None, None),
        n_points: int = 100,
        label: Optional[str] = "Emulator",
        blur: bool = False,
        color: str = digilab_colors["light_blue"],
        figsize: Tuple[float, float] = (6.4, 4.8),
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
                To pass through "None", either leave x_fixed out or pass through an empty dictionary.
            params: (PredictParams, optional). A parameter configuration that contains optional prediction parameters.
            x_lim ([tuple[Union[float, None], Union[float, None]]), optional]: The limits of the x-axis.
                If either is not provided. the limits will be taken directly from the emulator.
            n_points (int, optional): The number of points to sample in the x-axis.
            label (Union[str, None], optional): The label for the line in the plot legend. Defaults to "Emulator".
            blur (bool, optional): If ``True`` the emulator prediction will be blurred to visualize the uncertainty in the emulator.
                If ``False`` the mean and one and two standard deviation bands will be plotted,
                which encase 68% and 95% of the emulator prediction respectively.
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
        _, response = _api.get_emulator_summary(self.project_id, self.id)
        inputs = set(response["summary"]["data_diagnostics"]["inputs"].keys())
        outputs = set(response["summary"]["data_diagnostics"]["outputs"].keys())

        # Check function inputs
        fixed_inputs = set(x_fixed.keys())
        if x_axis not in inputs:
            raise ValueError(
                f"x_axis '{x_axis}' must be one of the emulator inputs: {inputs}"
            )
        if x_axis in fixed_inputs:
            raise ValueError(
                f"x_axis '{x_axis}' must not be one of the fixed emulator inputs: {fixed_inputs}"
            )
        if y_axis not in outputs:
            raise ValueError(
                f"y_axis '{y_axis}' must be one of the emulator outputs: {outputs}"
            )
        specified_inputs = fixed_inputs.union({x_axis})
        if inputs != specified_inputs:
            print(f"input columns: {inputs}")
            print(
                f"x_axis column: {set(x_axis)}"
            )  # 'set' here to harmoise output in these prints
            print(f"x_fixed columns: {fixed_inputs}")
            if len(specified_inputs) > len(inputs):
                error_text = (
                    "You have specified more columns than the emulator has as inputs."
                )
            elif len(specified_inputs) < len(inputs):
                error_text = "You have not specified all of the columns that the emulator has as inputs."
            else:
                error_text = "You have specified the wrong columns."
            raise ValueError(
                f"All input columns must be specified either as x_axis or in x_fixed (and not both). {error_text}"
            )

        # Get the range for the x-axis either from the user or from the emulator
        # if x_lim is not None:
        #     xmin, xmax = x_lim
        # else:
        #     inputs = response["summary"]["data_diagnostics"]["inputs"]
        #     xmin, xmax = inputs[x_axis]["min"], inputs[x_axis]["max"]
        xmin, xmax = x_lim
        if xmin is None or xmax is None:
            inputs = response["summary"]["data_diagnostics"]["inputs"]
        if xmin is None:
            xmin = inputs[x_axis]["min"]
        if xmax is None:
            xmax = inputs[x_axis]["max"]

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
        if blur:
            plotting_function = _plotting.blur
        else:
            plotting_function = _plotting.plot
        plt = plotting_function(
            x_axis,
            y_axis,
            df_X,
            df_mean,
            df_std,
            color=color,
            label=label,
            figsize=figsize,
        )
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
        x1_lim: Tuple[Optional[float], Optional[float]] = (None, None),
        x2_lim: Tuple[Optional[float], Optional[float]] = (None, None),
        y_lim: Tuple[Optional[float], Optional[float]] = (None, None),
        n_points: int = 25,
        cmap=digilab_cmap,  # NOTE: No typehint beacause the same as matplolib cmap (string & objects)?
        figsize: Tuple[float, float] = (6.4, 4.8),
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
            y_lim (tuple[float, float], optional): The limits of the y-axis; this is the colorbar.
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
        _, response = _api.get_emulator_summary(self.project_id, self.id)
        inputs = set(response["summary"]["data_diagnostics"]["inputs"].keys())
        outputs = set(response["summary"]["data_diagnostics"]["outputs"].keys())

        # Check function inputs
        fixed_inputs = set(x_fixed.keys())
        if x1_axis not in inputs:
            raise ValueError(
                f"x1_axis '{x1_axis}' must be one of the Emulator inputs: {inputs}"
            )
        if x1_axis in fixed_inputs:
            raise ValueError(
                f"x1_axis '{x1_axis}' must not be one of the fixed emulator inputs: {fixed_inputs}"
            )
        if x2_axis not in inputs:
            raise ValueError(
                f"x2_axis '{x1_axis}' must be one of the Emulator inputs: {inputs}"
            )
        if x2_axis in fixed_inputs:
            raise ValueError(
                f"x2_axis '{x2_axis}' must not be one of the fixed emulator inputs: {fixed_inputs}"
            )
        if y_axis not in outputs:
            raise ValueError(f"y_axis must be one of the Emulator outputs: {outputs}")
        specified_inputs = fixed_inputs.union({x1_axis, x2_axis})
        if inputs != specified_inputs:
            print(f"input columns: {inputs}")
            print(
                f"x1_axis column: {set(x1_axis)}"
            )  # 'set' here to harmoise output in these prints
            print(
                f"x2_axis column: {set(x2_axis)}"
            )  # 'set' here to harmoise output in these prints
            print(f"x_fixed columns: {fixed_inputs}")
            if len(specified_inputs) > len(inputs):
                error_text = (
                    "You have specified more columns than the emulator has as inputs."
                )
            elif len(specified_inputs) < len(inputs):
                error_text = "You have not specified all of the columns that the emulator has as inputs."
            else:
                error_text = "You have specified the wrong columns."
            raise ValueError(
                f"All input columns must be specified either as x1_axis, x2_axis, or in x_fixed (and not both). {error_text}"
            )

        # Get the ranges for the x-axes either from the user or from the emulator
        inputs = response["summary"]["data_diagnostics"]["inputs"]
        x1min, x1max = x1_lim
        x2min, x2max = x2_lim
        if x1min is None:
            x1min = inputs[x1_axis]["min"]
        if x1max is None:
            x1max = inputs[x1_axis]["max"]
        if x2min is None:
            x2min = inputs[x2_axis]["min"]
        if x2max is None:
            x2max = inputs[x2_axis]["max"]

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
        df_mean, df_std = self.predict(df_X, params=params, verbose=verbose)
        if mean_or_std == "mean":
            df = df_mean
        elif mean_or_std == "std":
            df = df_std
        else:
            raise ValueError("mean_or_std must be either 'mean' or 'std'")

        # Plot the results
        plt = _plotting.heatmap(
            x1_axis,
            x2_axis,
            y_axis,
            df_X,
            df,
            cmap,
            figsize,
            vmin=y_lim[0],
            vmax=y_lim[1],
        )
        return plt  # Return the plot

    @typechecked
    def export(
        self,
        file_path: str,
        format: str,
        observation_noise: Optional[bool] = None,
        verbose: bool = True,
    ) -> None:
        """
        Export your emulator using a valid file format.
        Currently twinLab support exporting emulators in the following formats:

        - ``"torchscript"``: Export the emulator as a TorchScript model for easy inference in PyTorch. Please see the `Pytorch documentation <https://pytorch.org/docs/stable/jit.html#>`_ for more information on how to use TorchScript models.

        Args:
            file_path (str): The local path to save the exported emulator.
            format (str): The format in which to export the emulator. Valid strings include ``"torchscript"``.
            observation_noise (bool, optional): If supported by your emulator, setting this to ``True`` means the emulator will be exported with observation noise.
            verbose (bool, optional): Display detailed information about the operation while running.

        Returns:
            None

        Examples:
            .. code-block:: python

                emulator = tl.Emulator("emulator_id")
                emulator.train(dataset, inputs, outputs, params)
                emulator.export("torchscript")
        """

        # Assert the format is in the valid enum
        try:
            format_enum = ValidExportFormats(format)
        except ValueError as e:
            raise ValueError(
                f"`'{format}'` is not a valid export format. Please choose from one of the following: {ValidExportFormats.list()}"
            ) from e

        if format_enum == ValidExportFormats.TORCHSCRIPT:

            # First check if the .pt file already exists

            _, response = _api.post_emulator_torchscript(
                self.project_id, self.id, {"observation_noise": observation_noise}
            )
            print(response)
            process_id = _utils.get_value_from_body("process_id", response)

            response = _utils.wait_for_job_completion(
                _api.get_emulator_torchscript,
                self.project_id,
                self.id,
                process_id,
                verbose=False,
            )

            result_url = response["result_url"]
            download_file_from_url(result_url, file_path)

        if verbose:
            print(f"Emulator exported to {file_path} successfully.")

    @typechecked
    def lock(self, verbose: bool = True) -> None:
        """Lock the emulator to prevent further training.

        This method will lock the emulator, preventing further training or deleting.

        Args:
            verbose (bool, optional): Display confirmation.

        Examples:
            .. code-block:: python

                emulator = tl.Emulator("emulator_id")
                emulator.lock()

        """
        _, response = _api.patch_emulator_lock(self.project_id, self.id)

        if verbose:
            detail = _utils.get_value_from_body("detail", response)
            print(detail)

    @typechecked
    def unlock(self, verbose: bool = True) -> None:
        """Unlock the emulator to allow further training.

        This method will unlock the emulator, allowing further training or deleting.

        Args:
            verbose (bool, optional): Display confirmation.

        Examples:
            .. code-block:: python

                emulator = tl.Emulator("emulator_id")
                emulator.unlock()

        """
        _, response = _api.patch_emulator_unlock(
            self.project_id,
            self.id,
        )

        if verbose:
            detail = _utils.get_value_from_body("detail", response)
            print(detail)

    @typechecked
    def fmu(
        self,
        file_path: str,
        states: dict,
        type: str = "model-exchange",
        os: str = "win64",
        verbose: bool = True,
    ) -> None:
        """
        Export your emulator as a Functional Mock-up Unit (FMU) following the FMI 2.0 standard.
        The FMU will be compatible for the specified operating system (by default, Windows 64-bit).
        Args:
            file_path (str): The path to save the exported emulator.
            states (dict): A dictionary mapping each input state to the corresponding output derivative, used to update inputs during simulations post-integration. Keys in the dictionary represent the input state names, while values specify the associated output derivative names.
            The dictionary length must match the number of emulator outputs, with each output assigned to one unique input. No output should be repeated within the mapping, ensuring comprehensive coverage of emulator outputs.
            type (str, optional): The type of FMU to export. Currently only ``"model-exchange"`` is supported.
            os (str, optional): The operating system to export the FMU for. Currently only ``"win64"`` is supported.
            verbose (bool, optional): Display detailed information about the operation while running.
        Returns:
            None
        Examples:
            .. code-block:: python
                emulator = tl.Emulator(id="emulator_name")
                emulator.train(dataset, inputs=["x1","x2","x3","x4"], outputs=["der(x1)","der(x2)"], params)
                emulator.fmu(file_path="emulator.fmu", states={"x1":"der(x1)", "x2":"der(x2)"})
        """

        # Assert the format is in the valid enum
        try:
            fmu_type_enum = ValidFMUTypes(type)
        except ValueError as e:
            raise ValueError(
                f"`'{type}'` is not a supported FMU type. Currently twinLab support the following formats: {ValidFMUTypes.list()}"
            ) from e

        try:
            os_enum = ValidFMUOS(os)
        except ValueError as e:
            raise ValueError(
                f"`'{os}'` is not currently supported. Please choose from one of the following operation systems: {ValidFMUOS.list()}"
            ) from e

        params = {"states": states}
        _, response = _api.post_emulator_fmu(self.id, params)
        process_id = _utils.get_value_from_body("process_id", response)

        response = _utils.wait_for_job_completion(
            _api.get_emulator_fmu, self.id, process_id, verbose=False
        )

        result_url = response["result_url"]
        download_file_from_url(result_url, file_path)

        if verbose:
            print(f"Emulator exported to {file_path} successfully.")
