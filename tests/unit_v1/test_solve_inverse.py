import io
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import twinlab


@pytest.fixture
def dfs():
    df_train = pd.DataFrame({"X": np.random.rand(3), "y": np.random.rand(3)})
    df_test = pd.DataFrame({"X": np.random.rand(3), "y": np.random.rand(3)})
    return df_train, df_test


@patch("twinlab.client._use_campaign")
def test_solve_inv_use(mock_use_campaign, dfs):
    """
    Test for twinlab.client.solve_inverse_campaign()

    Requirements
    ------------
        method calls _use_campaign
    """
    # Setup
    df_train, df_test = dfs

    with pytest.raises(TypeError):
        _ = twinlab.client.solve_inverse_campaign(
            campaign_id="solve-inverse",
            filepath_or_df=df_train,
            filepath_or_df_std=df_test,
            processor="cpu",
            verbose=False,
            debug=False,
        )

    mock_use_campaign.assert_called_once()


@patch("twinlab.client._use_campaign")
def test_solve_inv_out(mock_use_campaign, dfs):
    """
    Test for twinlab.client.solve_inverse_campaign()

    Requirements
    ------------
        method returns the correct (self-defined) DataFrame
    """
    # Setup
    df_train, df_test = dfs

    io_string = ",mean,sd,hdi_3%,hdi_97%,mcse_mean,mcse_sd,ess_bulk,ess_tail,r_hat\na,3.002,0.623,2.035,4.097,0.015,0.01,1853.0,3524.0,1.0"
    mock_use_campaign.return_value = io.StringIO(io_string)

    # Define the expected DataFrame
    expected_df = pd.read_csv(io.StringIO(io_string))
    expected_df = expected_df.set_index("Unnamed: 0")
    expected_df.index.name = None
    if "Unnamed: 0.1" in expected_df.columns:
        expected_df = expected_df.drop("Unnamed: 0.1", axis=1)

    result = twinlab.client.solve_inverse_campaign(
        campaign_id="solve-inverse",
        filepath_or_df=df_train,
        filepath_or_df_std=df_test,
        processor="cpu",
        verbose=False,
        debug=False,
    )

    pd.testing.assert_frame_equal(result, expected_df)
