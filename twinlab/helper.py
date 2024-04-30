import json
from pprint import pprint

import pandas as pd
from typeguard import typechecked


@typechecked
def load_dataset(filepath: str, verbose: bool = False) -> pd.DataFrame:
    """Load a dataset from a local file in ``.csv`` format into a pandas dataframe.

    Args:
        filepath (str): Path to the dataset file, which should be in csv format.
        verbose (bool, optional): Display information while running.

    Returns:
        pd.DataFrame: The dataset loaded from the file.

    Example:
        .. code-block:: python

            df = tl.load_dataset("path/to/data.csv", verbose=True)

        .. code-block:: console

            Dataset loaded:
                 x         y
            0  0.0  1.097485
            1  1.0  0.835439
            2  2.0  0.655124
    """
    df = pd.read_csv(filepath)
    if verbose:
        print("Dataset loaded:")
        print(df)
    return df


def load_params(filepath: str, verbose: bool = False) -> dict:
    """Load a parameter set from a local file in ``.json`` format into a dictionary.

    Args:
        filepath (str): Path to the dataset file, which should be in json format.
        verbose (bool, optional): Display information while running.

    Returns:
        dict: The parameter set loaded from the file.

    Example:

        .. code-block:: python

                params = tl.load_params("path/to/params.json", verbose=True)
    """

    with open(filepath) as f:
        params = json.load(f)
    if verbose:
        print("Parameters loaded from file:")
        pprint(params)
    return params


def get_sample(df: pd.DataFrame, key: int) -> pd.DataFrame:
    """Retrieve an individual sample from the multi-indexed dataframe returned by the ``Emulator.sample()`` method.

    The output from the ``Emulator.sample()`` method is a multi-indexed dataframe where the first level of the index is the parameter name and the second level is the sample number.
    This convenience method allows you to isolate an individual sample from the dataframe by providing the sample number.
    The sample is returned as a standard dataframe.

    Args:
        df (pd.DataFrame): A multi-indexed dataframe returned by the Emulator.sample() method.
        key (int): The integer key for the sample to retrieve.

    Returns:
        pd.DataFrame: The individual sample as a standard (non-multi-indexed) dataframe.

    Example:
        .. code-block:: python

            df = emulator.sample(df_X, 5) # Generates independent samples
            tl.get_sample(df, 1) # Isolate sample "1"

        .. code-block:: console

                      y
            0  1.097485
            1  0.835439
            2  0.655124

    """
    key_str = str(key)
    sample_df = df.xs(key=key_str, level=-1, axis="columns")
    return sample_df


def join_samples(
    df_one: pd.DataFrame,
    df_two: pd.DataFrame,
) -> pd.DataFrame:
    """Join two dataframes that contain independent samples generated by the ``Emulator.sample()`` method.

    The output from the ``Emulator.sample()`` method is a multi-indexed dataframe where the first level of the index is the parameter name and the second level is the sample number.
    This convenience method allows you to join two dataframes that contain independent samples generated by the ``Emulator.sample()`` method together into a single dataframe.

    Args:
        df_one (pd.DataFrame): The first multi-indexed dataframe to join.
        df_two (pd.DataFrame): The second multi-indexed dataframe to join.

    Returns:
        pd.DataFrame: The joined dataframe

    Example:
        .. code-block:: python

            df_y1 = emulator.sample(df_X, 1) # Create first set of samples
            df_y2 = emulator.sample(df_X, 3) # Creates new independent samples
            tl.join_samples(df_y1, df_y2)

        .. code-block:: console

                      y
                      0         1         2         3
            0  0.784193  1.308067  0.176582  0.875387
            1  0.978259  1.039125  0.646922  0.887118
            2  1.086855  0.942270  0.864730  0.934348
    """
    # Get the number of samples in the first dataframe to offset the indices of the second dataframe
    n_samples = int(len(df_one.columns) / len(df_one.columns.levels[0]))

    # Convert the second dataframe to a dictionary
    df_dict = df_two.to_dict()
    new_dict, new_keys = {}, {}

    # Form the new dictionary with updated indices
    for key in df_dict.keys():
        _key = (key[0], str(int(key[1]) + n_samples))
        new_keys[key] = _key
    for key in new_keys:
        new_dict[new_keys[key]] = df_dict[key]

    # Convert the new dictionary to a dataframe
    new_df = pd.DataFrame(new_dict)
    sorted_df = pd.concat([df_one, new_df], axis=1).sort_index(axis=1, level=0)[
        df_one.columns.get_level_values(0).unique()
    ]
    return sorted_df