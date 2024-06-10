# Standard imports
import io
from typing import List

import pandas as pd

# Third-party imports
from deprecated import deprecated
from typeguard import typechecked

# Project imports
from . import api, settings, utils

# Parameters
DEBUG = False  # For developer debugging purposes
USE_UPLOAD_URL = True  # Needs to be set to True to allow for large dataset uploads
DEPRECATION_VERSION = "2.5.0"
DEPRECATION_MESSAGE = (
    "This method is being deprecated. Please use `Dataset.analyse_variance()` to analyse either input or output variance.",
)


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

    def __init__(self, id: str):
        self.id = id

    def __str__(self):
        return f"Dataset ID: {self.id}"

    @typechecked
    def upload(
        self,
        df: pd.DataFrame,
        verbose: bool = False,
    ) -> None:
        """Upload a dataset to the twinLab cloud so that it can be queried and used for training.

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
            _, response = api.generate_upload_url(self.id, verbose=DEBUG)
            upload_url = utils.get_value_from_body("url", response)
            utils.upload_dataframe_to_presigned_url(
                df,
                upload_url,
                verbose=verbose,
                check=settings.CHECK_DATASETS,
            )
            if verbose:
                print("Processing dataset")
            _, response = api.process_uploaded_dataset(self.id, verbose=DEBUG)
        else:
            csv_string = utils.get_csv_string(df)
            _, response = api.upload_dataset(self.id, csv_string, verbose=DEBUG)
        if verbose:
            message = utils.get_message(response)
            print(message)

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
        columns_string = ",".join(columns)
        _, response = api.analyse_dataset(
            self.id, columns=columns_string, verbose=DEBUG
        )
        if response["dataframe"] is not None:
            csv_string = utils.get_value_from_body("dataframe", response)
            csv_string = io.StringIO(csv_string)
        else:
            df_url = utils.get_value_from_body("dataframe_url", response)
            csv_string = utils.download_dataframe_from_presigned_url(df_url)
            dataframe_name = utils.get_value_from_body("dataframe_name", response)
            _, response = api.delete_temp_dataset(dataframe_name)
        df = pd.read_csv(csv_string, index_col=0, sep=",")
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
        _, response = api.view_dataset(self.id, verbose=DEBUG)
        if response["dataset"] is not None:
            csv_string = utils.get_value_from_body("dataset", response)
            csv_string = io.StringIO(csv_string)
        else:
            df_url = utils.get_value_from_body("dataset_url", response)
            csv_string = utils.download_dataframe_from_presigned_url(df_url)
        df = pd.read_csv(csv_string, sep=",")
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
        _, response = api.summarise_dataset(self.id, verbose=DEBUG)

        csv_string = utils.get_value_from_body("dataset_summary", response)
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
        _, response = api.delete_dataset(self.id, verbose=DEBUG)

        if verbose:
            message = utils.get_message(response)
            print(message)
