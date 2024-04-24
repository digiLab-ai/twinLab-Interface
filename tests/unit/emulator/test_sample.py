from unittest.mock import patch

import pandas as pd
import pytest
import twinlab as tl


@pytest.fixture
def sample_test_df():
    return pd.DataFrame(
        {
            "Sunlight [hours/day]": [6, 8, 10, 9, 7],
            "Water [times/week]": [2.0, 3.0, 3.0, 4.0, 2.0],
        }
    )


@pytest.fixture
def mock_sample_request_model_success():
    return 200, {
        "message": "Campaign method sample started",
        "process_id": "b09177b5-f518-453e-8023-0277ed00eeae",
    }


@pytest.fixture
def mock_sample_response_model_success():
    return 200, {
        "dataframe": "Fruits produced,Fruits produced,Fruits produced\n0,1,2\n6.859549750072295,5.742606302334472,4.59464930514722\n6.386964718351616,4.253101167743323,3.822612804047851\n5.968952226622479,4.37671999746412,2.6158350758145956\n4.4926251773876436,3.723341353096911,4.380736519327339\n5.518459790402031,5.234296216843149,3.394593816401424\n",
        "process_id": "b09177b5-f518-453e-8023-0277ed00eeae",
        "process_status": "done_Modal",
    }


@patch("twinlab.api.use_response_model")
@patch("twinlab.api.use_request_model")
def test_sample(
    mock_sample_request_model,
    mock_sample_response_model,
    mock_sample_request_model_success,
    mock_sample_response_model_success,
    sample_test_df,
    dataframe_regression,
):

    # Arange
    emulator = tl.Emulator(id="test_emulator")
    mock_sample_request_model.return_value = mock_sample_request_model_success
    mock_sample_response_model.return_value = mock_sample_response_model_success

    # Act
    samples = emulator.sample(sample_test_df, 3)

    # Assert
    dataframe_regression.check(samples)
