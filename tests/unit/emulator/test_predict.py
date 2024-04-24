from unittest.mock import patch

import pandas as pd
import pytest
import twinlab as tl


@pytest.fixture
def predict_test_df():
    return pd.DataFrame(
        {
            "Sunlight [hours/day]": [6, 8, 10, 9, 7],
            "Water [times/week]": [2.0, 3.0, 3.0, 4.0, 2.0],
        }
    )


@pytest.fixture
def mock_predict_request_model_success():
    return 200, {
        "message": "Campaign method predict started",
        "process_id": "b09177b5-f518-453e-8023-0277ed00eeae",
    }


@pytest.fixture
def mock_predict_response_model_success():
    return 200, {
        "dataframe": "Fruits produced,Fruits produced [std_dev]\n5.921269093761975,0.9343242696122288\n5.638325090044324,1.4569355522854592\n4.947484126166849,1.7192462004496627\n4.703377413111889,1.2963723683471289\n5.0356945854066035,0.9475269135541184\n",
        "process_id": "b09177b5-f518-453e-8023-0277ed00eeae",
        "process_status": "done_Modal",
    }


@patch("twinlab.api.use_response_model")
@patch("twinlab.api.use_request_model")
def test_predict(
    mock_predict_request_model,
    mock_predict_response_model,
    mock_predict_request_model_success,
    mock_predict_response_model_success,
    predict_test_df,
    dataframe_regression,
):

    # Arange
    emulator = tl.Emulator(id="test_emulator")
    mock_predict_request_model.return_value = mock_predict_request_model_success
    mock_predict_response_model.return_value = mock_predict_response_model_success

    # Act
    df_mean, df_std = emulator.predict(predict_test_df)

    # Assert
    dataframe_regression.check(df_mean, basename="predict_mean")
    dataframe_regression.check(df_std, basename="predict_std")
