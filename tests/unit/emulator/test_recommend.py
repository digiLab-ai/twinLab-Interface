from unittest.mock import patch

import pytest
import twinlab as tl


@pytest.fixture
def mock_recommend_request_model_success():
    return 200, {
        "message": "Campaign method recommend started",
        "process_id": "b09177b5-f518-453e-8023-0277ed00eeae",
    }


@pytest.fixture
def mock_recommend_response_model_success():
    return 200, {
        "acq_func_value": "1",
        "dataframe": "Sunlight [hours/day],Water [times/week]\n6.432401475850908,1.5072378435806058\n4.420062807781083,2.373036349339225\n9.89330249240159,3.434895679896722\n",
        "process_id": "b09177b5-f518-453e-8023-0277ed00eeae",
        "process_status": "done_Modal",
    }


@patch("twinlab.api.use_response_model")
@patch("twinlab.api.use_request_model")
def test_recommend(
    mock_recommend_request_model,
    mock_recommend_response_model,
    mock_recommend_request_model_success,
    mock_recommend_response_model_success,
    dataframe_regression,
):

    # Arange
    emulator = tl.Emulator(id="test_emulator")
    mock_recommend_request_model.return_value = mock_recommend_request_model_success
    mock_recommend_response_model.return_value = mock_recommend_response_model_success

    # Act
    df, _ = emulator.recommend(3, "EI")

    # Assert
    dataframe_regression.check(df)
