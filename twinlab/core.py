import io
import os
from pprint import pprint
from typing import Dict, List, Optional

import pandas as pd
from typeguard import typechecked

from . import api, utils

# Parameters
DEBUG = False  # NOTE: For developer debugging purposes, turn to True to see more verbosity and logging from the API


@typechecked
def set_api_key(api_key: str, verbose: bool = False) -> None:
    """Set the user API key for their twinLab cloud account.

    Setting this will override the API key set in the environment variable ``TWINLAB_API_KEY`` in the .env file for the current session.
    This can be used instead of setting the API key in the ``.env`` file and negates the need for a ``.env`` file.

    Args:
        api_key (str): API key to use for access to the twinLab cloud.
        verbose (bool, optional): Display information about the operation while running.

    Example:
        .. code-block:: python

            tl.set_api_key("12345")

    """
    os.environ["TWINLAB_API_KEY"] = api_key
    if verbose:
        print("API key: {}".format(api_key))


@typechecked
def set_server_url(url: str, verbose: bool = False) -> None:
    """Set the server URL for twinLab.

    If this is not set the default URL is used: ``https://twinlab.digilab.co.uk``.
    This default URL will be correct for most users and should not normally need to be changed.
    Setting this will override the URL set in the environment variable ``TWINLAB_URL`` in the ``.env`` file for the current session.

    Args:
        url (str): URL for the twinLab cloud.
        verbose (bool, optional): Display information about the operation while running.

    Example:
        .. code-block:: python

            tl.set_server_url("https://twinlab.digilab.co.uk/stage")

    """
    os.environ["TWINLAB_URL"] = url
    if verbose:
        print("Server URL: {}".format(url))


@typechecked
def get_server_url(verbose: bool = False) -> str:
    """Show the URL from which twinLab is currently being accessed.

    Args:
        verbose (bool, optional): Display information about the operation while running.

    Returns:
        str: Server URL.

    Example:
        .. code-block:: python

            tl.get_server_url()

        .. code-block:: console

            'https://twinlab.digilab.co.uk'

    """
    server_url = os.getenv("TWINLAB_URL")
    if verbose:
        print("Server URL: {}".format(server_url))
    return server_url


@typechecked
def get_api_key(verbose: bool = False) -> str:
    """Show the user API key for their twinLab cloud account.

    Args:
        verbose (bool, optional): Display information about the operation while running.

    Returns:
        str: User API key.

    Example:
        .. code-block:: python

            tl.get_api_key()

        .. code-block:: console

            'secret-12345'

    """
    api_key = os.getenv("TWINLAB_API_KEY")
    if verbose:
        print("API key: {}".format(api_key))
    return api_key


@typechecked
def user_information(verbose: bool = False) -> Dict:
    """Get information about the twinLab user.

    Args:
        verbose (bool, optional): Display information about the operation while running.

    Returns:
        dict: User information.

    Example:
        .. code-block:: python

            user_info = tl.user_information()
            print(user_info)

        .. code-block:: python

            {'username': 'dodders@digilab.co.uk', 'credits': 850}

    """
    _, response = api.get_user(verbose=DEBUG)
    user_info = {}
    user_info["User"] = response.get("username")
    if verbose:
        print("User information:")
        pprint(user_info, compact=True, sort_dicts=False)
    return user_info


@typechecked
def versions(verbose: bool = False) -> Dict[str, str]:
    """Get information about the twinLab version being used.

    Args:
        verbose (bool, optional): Display information about the operation while running.

    Returns:
        dict: Version information.

    Example:
        .. code-block:: python

            version_info = tl.versions()
            print(version_info)

        .. code-block:: console

            {'cloud': '2.3.0', 'modal': '0.4.0', 'library': '1.7.0', 'image': 'twinlab'}

    """
    _, response = api.get_versions(verbose=DEBUG)
    version_info = response
    if verbose:
        print("Version information:")
        pprint(version_info, compact=True, sort_dicts=False)
    return version_info


@typechecked
def list_datasets(verbose: bool = False) -> List[str]:
    """List datasets that have been uploaded to the user's twinLab cloud account.

    These datasets can be used for training emulators and for other operations.
    New datasets can be uploaded using the ``upload`` method of the ``Dataset`` class.

    Args:
        verbose (bool, optional): Display information about the operation while running.

    Returns:
        list: Datasets currently uploaded to the user's `twinLab` cloud account.

    Example:
        .. code-block:: python

            datasets = tl.list_datasets()
            print(datasets)

        .. code-block:: console

            ['biscuits', 'gardening', 'force-energy', 'combusion']

    """
    _, response = api.list_datasets(verbose=DEBUG)
    datasets = utils.get_value_from_body("datasets", response)
    if verbose:
        print("Datasets:")
        pprint(datasets, compact=True, sort_dicts=False)
    return datasets


@typechecked
def list_emulators(
    verbose: bool = False,
) -> List[str]:
    """List trained emulators that exist in the user's twinLab cloud account.

    These trained emulators can be used for a variety of inference operations (see methods of the Emulator class).
    Setting ``verbose=True`` will also show the state of currently training emulators, as well as those that have failed to train.
    Information about start time and current/final run time will also be shown if ``verbose=True``.

    Args:
        verbose (bool, optional): Display information about the emulators while running.

    Returns:
        list: Currently trained emulators.

    Example:
        .. code-block:: python

            emulators = tl.list_emulators()
            print(emulators)

        .. code-block:: console

            ['biscuits', 'gardening', 'new-emulator', 'my-emulator']

    """

    # Get the emulators dictionary from the response
    _, response = api.list_models(verbose=DEBUG)
    emulators = utils.get_value_from_body("models", response)

    # Print detailed emulator information to screen
    if verbose:

        # Create dictionary of cuddly status messages
        status_messages = {
            "success": "Successfully trained emulators:",
            "in progress": "Emulators currently training:",  # NOTE: No underscore here
            "failed": "Emulators that failed to train:",
            "no_logging": "Emulators trained prior to logging:",  # NOTE: Underscore here
            "?": "Emulators with unknown status:",
        }

        # Get the detailed model information and sort by "start_time"
        # NOTE: Ensure that "start_time" is present in the dictionary
        emulator_info = utils.get_value_from_body("model_information", response)
        for emulator in emulator_info:
            if not emulator.get("start_time"):
                emulator["start_time"] = "N/A"
        emulator_info = sorted(emulator_info, key=lambda d: d["start_time"])

        # Sort emulators by status
        status_emulators = {status: [] for status in status_messages.keys()}
        for emulator in emulator_info:
            status = emulator.pop("status", None)
            if status in status_messages.keys():
                status_emulators[status].append(emulator)
            else:
                status_emulators["?"].append(emulator)

        # Print to screen
        for status, message in status_messages.items():
            if status_emulators[status]:
                print(message)
                pprint(status_emulators[status], compact=True, sort_dicts=False)
                print()

    return emulators


@typechecked
def list_example_datasets(verbose: bool = False) -> list:
    """List example datasets that are available on the twinLab cloud.

    These datasets can be downloaded and used for training emulators and other operations and are used in many of the examples in the documentation.
    Datasets can be loaded using the ``load_example_dataset`` function.

    Args:
        verbose (bool, optional): Display information about the operation while running.

    Returns:
        list: Example datasets available for loading.

    Example:
        .. code-block:: python

            example_datasets = tl.list_example_datasets()
            print(example_datasets)

        .. code-block:: console

            ['biscuits', 'gardening', 'quickstart', 'tritium-desorption']

    """
    _, response = api.list_example_datasets(verbose=verbose)
    datasets = utils.get_value_from_body("datasets", response)
    if verbose:
        print("Example datasets:")
        pprint(datasets, compact=True, sort_dicts=False)

    return datasets


def load_example_dataset(dataset_id: str, verbose: bool = False) -> pd.DataFrame:
    """Load an example dataset from the twinLab cloud into the current session.

    Args:
        dataset_id (str): The name of the dataset to load.
        verbose (bool, optional): Display information about the operation while running.

    Returns:
        pandas.DataFrame: The example dataset.

    Example:
        .. code-block:: python

            tl.load_example_dataset("quickstart")

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
    _, response = api.load_example_dataset(dataset_id, verbose=verbose)
    csv = utils.get_value_from_body("dataset", response)
    csv = io.StringIO(csv)
    df = pd.read_csv(csv, sep=",")
    if verbose:
        print("Example dataset:")
        print(df)
        print("Dataframe downloaded successfully.")
    return df
