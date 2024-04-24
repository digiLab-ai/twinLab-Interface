from unittest.mock import patch

import pytest
import twinlab as tl


@pytest.fixture
def mock_list_process_response():
    return 200, {
        "processes": [
            {
                "process_id": "69f25ac8-ab73-4067-a625-8f0183f1840d",
                "method": "predict",
                "run-time": "0:00:06",
                "start_time": "2024-03-18 15:13:47",
                "status": "success",
            },
            {
                "process_id": "66022a34-574d-4207-acd5-24a2e23b7bd3",
                "method": "sample",
                "run-time": "0:00:06",
                "start_time": "2024-03-18 15:13:47",
                "status": "success",
            },
        ],
    }


@pytest.fixture
def mock_use_response_model_predict_response():
    return 200, {
        "process_id": "69f25ac8-ab73-4067-a625-8f0183f1840d",
        "process_status": "Your job has finished and is on it's way back to you",
        "dataframe": "Fruits produced,Fruits produced [std_dev]\n5.921269093761975,0.9343242696122288\n5.638325090044324,1.4569355522854592\n4.947484126166849,1.7192462004496627\n4.703377413111889,1.2963723683471289\n5.0356945854066035,0.9475269135541184\n",
    }


@pytest.fixture
def mock_use_response_model_other_response():
    return 200, {
        "process_id": "66022a34-574d-4207-acd5-24a2e23b7bd3",
        "process_status": "Your job has finished and is on it's way back to you",
        "dataframe": "Fruits produced,Fruits produced,Fruits produced\n0,1,2\n2.5072915449742634,-0.8628879986709295,-4.32664544918507\n3.888081135302346,0.313525414622509,-0.3971001985085243\n",
    }


@patch("twinlab.api.use_response_model")
@patch("twinlab.api.list_processes_model")
def test_get_predict_process(
    mock_list_process_model,
    mock_use_response_model,
    mock_list_process_response,
    mock_use_response_model_predict_response,
    dataframe_regression,
):
    # Arange
    emulator = tl.Emulator(id="test_emulator")
    process_id = mock_use_response_model_predict_response[1]["process_id"]
    mock_list_process_model.return_value = mock_list_process_response
    mock_use_response_model.return_value = mock_use_response_model_predict_response

    # Act
    df_mean, df_std = emulator.get_process(process_id)

    # Assert
    dataframe_regression.check(df_mean, basename="get_process_predict_mean")
    dataframe_regression.check(df_std, basename="get_process_predict_std")


@patch("twinlab.api.use_response_model")
@patch("twinlab.api.list_processes_model")
def test_get_other_process(
    mock_list_process_model,
    mock_use_response_model,
    mock_list_process_response,
    mock_use_response_model_other_response,
    dataframe_regression,
):
    # Arange
    emulator = tl.Emulator(id="test_emulator")
    process_id = mock_use_response_model_other_response[1]["process_id"]
    mock_list_process_model.return_value = mock_list_process_response
    mock_use_response_model.return_value = mock_use_response_model_other_response

    # Act
    df = emulator.get_process(process_id)

    # Assert
    dataframe_regression.check(df, basename="get_other_process")
