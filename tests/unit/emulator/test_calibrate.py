from unittest.mock import patch

import pandas as pd
import pytest
import twinlab as tl


@pytest.fixture
def calibrate_test_dfs():
    return pd.DataFrame(
        {
            "Fruit Produced": [10],
        }
    ), pd.DataFrame(
        {
            "Fruit Produced": [1],
        }
    )


@pytest.fixture
def mock_calibrate_request_model_success():
    return 200, {
        "message": "Campaign method calibrate started",
        "process_id": "57bca4b0-609b-41ee-9534-4af7284a028f",
    }


@pytest.fixture
def mock_calibrate_response_model_success():
    return 200, {
        "dataframe": ",mean,sd,hdi_3%,hdi_97%,mcse_mean,mcse_sd,ess_bulk,ess_tail,r_hat\nSunlight [hours/day],10.031,1.273,8.685,11.346,0.833,0.684,3.0,16.0,2.32\nWater [times/week],2.151,2.058,0.003,4.299,1.347,1.106,3.0,18.0,2.81\n",
        "process_id": "57bca4b0-609b-41ee-9534-4af7284a028f",
        "process_status": "done_Modal",
    }


@patch("twinlab.api.use_response_model")
@patch("twinlab.api.use_request_model")
def test_calibrate(
    mock_calibrate_request_model,
    mock_calibrate_response_model,
    mock_calibrate_request_model_success,
    mock_calibrate_response_model_success,
    calibrate_test_dfs,
    dataframe_regression,
):

    # Arange
    emulator = tl.Emulator(id="test_emulator")
    mock_calibrate_request_model.return_value = mock_calibrate_request_model_success
    mock_calibrate_response_model.return_value = mock_calibrate_response_model_success

    # Act
    df = emulator.calibrate(*calibrate_test_dfs)

    # Assert
    dataframe_regression.check(df)
