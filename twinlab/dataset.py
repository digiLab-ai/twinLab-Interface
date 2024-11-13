# Standard imports
import io
import os
import uuid
from typing import List

import pandas as pd

# Third-party imports
from deprecated import deprecated
from typeguard import typechecked

# Project imports
from . import _api, _utils, settings

# Parameters
DEBUG = False  # For developer debugging purposes
USE_UPLOAD_URL = True  # Needs to be set to True to allow for large dataset uploads
DEPRECATION_VERSION = "2.5.0"
DEPRECATION_MESSAGE = (
    "This method is being deprecated. Please use `Dataset.analyse_variance()` to analyse either input or output variance.",
)


def _process_request_dataframes(project_id: str, df: pd.DataFrame) -> str:
    # Create a unique ID for the dataset
    dataset_id = str(uuid.uuid4())

    # Generate a temporary upload URL - this URL does not have the 5.5mb limit
    _, response = _api.get_dataset_temporary_upload_url(project_id, dataset_id)
    url = _utils.get_value_from_body("url", response)

    # Upload the dataframe to the presigned URL
    _utils.upload_dataframe_to_presigned_url(df, url, check=settings.CHECK_DATASETS)

    return dataset_id


class Dataset:
    """A twinLab dataset that can be used for training an emulator online.

    Note that instantiating a dataset object locally does not create a new dataset on the twinLab cloud.
    Instead, it can be used either to interact with an existing dataset that has been uploaded to the cloud or as a precursor step to uploading a new dataset.

    Args:
        id (str): Name of the dataset.

    Example:
        .. code-block:: python

            dataset = tl.Dataset("my_dataset")

    """

    def __init__(
        self,
        id: str,
        project: str = "personal",
        project_owner: str = None,
    ):
        self.id = id
        self.project_id = _utils.match_project(project, project_owner)

    def __str__(self):
        return f"Dataset ID: {self.id}"

    @typechecked
    def upload(
        self,
        df: pd.DataFrame,
        verbose: bool = False,
    ) -> None:
        """Upload a dataset to the twinLab cloud so that it can be queried and used for training.

        If a dataset has been uploaded previously for this Dataset object, this will require you to re-instantiate the dataset object with a new ID (`tl.Dataset("new_id")`) or delete the existing dataset object (`Dataset.delete`) before proceeding.

        Please note the largest dataset that can be uploaded is currently 5GB.

        When using twinLab emulators, note that training time scales cubically with the amount of data included.
        It may be worthwhile training with a smaller subset of data at first, to determine approximately how long it will take to train.
        Please get in touch with our experts for technical support to understand how to make best use of data.

        Args:
            df (pandas.DataFrame): A `pandas.DataFrame` containing the dataset to be uploaded.
            verbose (bool, optional): Display information about the operation while running.

        Example:
            .. code-block:: python

                dataset = tl.Dataset("my_dataset")
                df = pd.DataFrame({"X": [1, 2, 3, 4], "y": [1, 4, 9, 16]})
                dataset.upload(df)

        """
        # Upload the file (either via link or directly)
        if USE_UPLOAD_URL:
            _, response = _api.get_dataset_upload_url(self.project_id, self.id)
            upload_url = _utils.get_value_from_body("url", response)
            _utils.upload_dataframe_to_presigned_url(
                df,
                upload_url,
                verbose=verbose,
                check=settings.CHECK_DATASETS,
            )
            response = _api.post_dataset_record(self.project_id, self.id)
        else:
            csv_string = _utils.get_csv_string(df)
            _, response = _api.post_dataset(self.project_id, self.id, csv_string)
            if verbose:
                detail = _utils.get_value_from_body("detail", response)
                print(detail)

        # Create the dataset summary
        _, response = _api.post_dataset_summary(self.project_id, self.id)
        if verbose:
            detail = _utils.get_value_from_body("detail", response)
            print(detail)

    @typechecked
    def append(
        self,
        df: pd.DataFrame,
        verbose: bool = False,
    ) -> None:
        """Append new data to an existing dataset in the twinLab cloud.

        Appending data to an existing dataset is useful when new data becomes available and needs to be added to the dataset for training purposes.
        This can be useful when new data is generated over time, for example when using twinLab's ``Emulator.recommend`` functionality.
        This method allows for the dataset to be updated directly on the cloud without needing to download, update and re-upload the entire dataset.

        Please note that the new dataset must have the same columns as the original dataset.

        Args:
            df (pandas.DataFrame): A `pandas.DataFrame` containing the dataset to be appended.
            verbose (bool, optional): Display information about the operation while running.

        Example:
            .. code-block:: python

                dataset = tl.Dataset("my_dataset")
                df = pd.DataFrame({"X": [1, 2, 3, 4], "y": [1, 4, 9, 16]})
                dataset.append(df)

        """
        # Do the appending
        dataset_id = _process_request_dataframes(self.project_id, df)
        _, response = _api.post_dataset_append(
            self.project_id,
            self.id,
            dataset_id,
        )
        if verbose:
            detail = _utils.get_value_from_body("detail", response)
            print(detail)

        # Update the dataset summary
        _, response = _api.post_dataset_summary(self.project_id, self.id)
        if verbose:
            detail = _utils.get_value_from_body("detail", response)
            print(detail)

    @typechecked
    def analyse_variance(
        self,
        columns: List[str],
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Return an analysis of the variance retained per dimension after performing singular value decomposition (SVD) on the dataset.

        SVD is useful for understanding how much variance in the dataset is retained after projecting it into a new basis.
        SVD components are naturally ordered by the amount of variance they retain, with the first component retaining the most variance.
        A decision can be made about how many dimensions to keep based on the cumulative variance retained.
        This analysis is usually performed on either the set of input or output columns of the dataset.

        Args:
            columns (list[str]): List of columns to evaluate. This is typically either the set of input or output columns.
            verbose (bool, optional): Display information about the operation while running.

        Returns:
            pandas.Dataframe: A ``pandas.DataFrame`` containing the variance analysis.

        Example:
            .. code-block:: python

                dataset = tl.Dataset("quickstart")
                dataset.analyse_variance(columns=["x", "y"]) # Typically either input or output columns

            .. code-block:: console

                   Number of Dimensions  Cumulative Variance
                0                     0             0.000000
                1                     1             0.925741
                2                     2             1.000000

        """
        if len(columns) == 1:
            raise ValueError(
                "Singular value decomposition should use more than one column. Please retry with more than one input or output column."
            )

        _, response = _api.post_dataset_analysis(self.project_id, self.id, columns)

        process_id = _utils.get_value_from_body("process_id", response)

        response = _utils.wait_for_job_completion(
            _api.get_dataset_process,
            self.project_id,
            self.id,
            process_id,
            verbose=verbose,
        )

        response = _utils.process_result_response(response)

        result = _utils.get_value_from_body("dataset_variance", response)

        df = pd.read_csv(io.StringIO(result), sep=",")

        if verbose:
            print("Variance Analysis:")
            print(df)
        return df

    @deprecated(
        version=DEPRECATION_VERSION,
        reason=DEPRECATION_MESSAGE,
    )
    @typechecked
    def analyse_input_variance(
        self,
        columns: List[str],
        verbose: bool = False,
    ) -> pd.DataFrame:
        """

        .. deprecated:: 2.5.0
            This method is being deprecated. Please use the method ``Dataset.analyse_variance()`` to analyse input or output variance.

        """
        df = self.analyse_variance(columns=columns, verbose=verbose)
        return df

    @deprecated(
        version=DEPRECATION_VERSION,
        reason=DEPRECATION_MESSAGE,
    )
    @typechecked
    def analyse_output_variance(
        self,
        columns: List[str],
        verbose: bool = False,
    ) -> pd.DataFrame:
        """

        .. deprecated:: 2.5.0
            This method is being deprecated. Please use the method ``Dataset.analyse_variance()`` to analyse input or output variance.

        """
        df = self.analyse_variance(columns=columns, verbose=verbose)
        return df

    # TODO: This should possibly be called 'download' instead of 'view'
    @typechecked
    def view(self, verbose: bool = False) -> pd.DataFrame:
        """View (and download) a dataset that exists on the twinLab cloud.

        Args:
            verbose (bool, optional): Display information about the operation while running.

        Returns:
            pandas.Dataframe: A ``pandas.DataFrame`` containing the requested dataset.

        Example:
            .. code-block:: python

                dataset = tl.Dataset("quickstart")
                dataset.view()

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
                8  0.480932  0.340115
                9  0.392118  0.845795

        """
        _, response = _api.get_dataset(self.project_id, self.id)
        df = _utils.process_dataset_response(response)
        if verbose:
            print("Dataset:")
            print(df)
        return df

    @typechecked
    def summarise(self, verbose: bool = False) -> pd.DataFrame:
        """Show summary statistics for a dataset that exists on the twinLab cloud.

        Args:
            verbose (bool, optional): Display information about the operation while running.

        Returns:
            pandas.DataFrame: A ``pandas.DataFrame`` with summary statistics.

        Example:
            .. code-block:: python

                dataset = tl.Dataset("my_dataset")
                dataset.summarise()

            .. code-block:: console

                               x          y
                count  10.000000  10.000000
                mean    0.544199   0.029383
                std     0.229352   0.748191
                min     0.226851  -0.960764
                25%     0.399865  -0.694614
                50%     0.516123   0.087574
                75%     0.693559   0.734513
                max     0.980764   0.921553

        """
        _, response = _api.get_dataset_summary(self.project_id, self.id)

        csv_string = _utils.get_value_from_body("summary", response)
        csv_string = io.StringIO(csv_string)
        df = pd.read_csv(csv_string, index_col=0, sep=",")
        if verbose:
            print("Dataset summary:")
            print(df)
        return df

    @typechecked
    def delete(self, verbose: bool = False) -> None:
        """Delete a dataset that was previously uploaded to the twinLab cloud.

        It can be useful to delete an emulator to keep a user's cloud account tidy, or if dataset has been set up incorrectly and no longer needs to be used.

        Args:
            verbose (bool, optional): Display information about the operation while running.

        Example:
            .. code-block:: python

                dataset = tl.Dataset("quickstart")
                dataset.delete()

        """
        _, response = _api.delete_dataset(self.project_id, self.id)

        if verbose:
            detail = _utils.get_value_from_body("detail", response)
            print(detail)

    @typechecked
    def lock(self, verbose: bool = False) -> None:
        """Lock a dataset that was previously uploaded to the twinLab cloud.

        Locking a dataset prevents it from being deleted or modified.

        Args:
            verbose (bool, optional): Display confirmation.

        Example:
            .. code-block:: python

                dataset = tl.Dataset("quickstart")
                dataset.lock()

        """
        _, response = _api.patch_dataset_lock(self.project_id, self.id)

        if verbose:
            detail = _utils.get_value_from_body("detail", response)
            print(detail)

    @typechecked
    def unlock(self, verbose: bool = False) -> None:
        """Unlock a dataset that was previously uploaded to the twinLab cloud.

        Unlocking a dataset allows it to be deleted or modified.

        Args:
            verbose (bool, optional): Display confirmation.

        Example:
            .. code-block:: python

                dataset = tl.Dataset("quickstart")
                dataset.unlock()

        """
        _, response = _api.patch_dataset_unlock(self.project_id, self.id)

        if verbose:
            detail = _utils.get_value_from_body("detail", response)
            print(detail)
