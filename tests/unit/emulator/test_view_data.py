from unittest.mock import patch

import pytest
import twinlab as tl


@pytest.fixture
def mock_view_training_data_response():
    return 200, {
        "training_data": ",x,y\n0,0.6964691855978616,-0.8173739564129022\n1,0.2861393349503794,0.8876561174050408\n2,0.2268514535642031,0.921552660721474\n3,0.5513147690828912,-0.3263338765412979\n4,0.7194689697855631,-0.8325176123242133\n5,0.4231064601244609,0.4006686354731812\n6,0.9807641983846156,-0.1649662650236807\n7,0.6848297385848633,-0.9607643657025954\n8,0.4809319014843609,0.3401149876855609\n9,0.3921175181941505,0.8457949914442409\n"
    }


@pytest.fixture
def mock_view_test_data_reponse():
    return 200, {
        "test_data": ",Sunlight [hours/day],Water [times/week],Fruits produced\n0,8.0,6.0,0.0\n1,0.2,3.0,0.0\n2,12.4,5.0,2.0\n3,14.1,0.0,0.0\n4,5.8,1.0,1.0\n"
    }


@patch("twinlab.api.view_data_model")
def test_view_train_data(
    mock_view_data_model, mock_view_training_data_response, dataframe_regression
):
    # Arange
    emulator = tl.Emulator(id="test_emulator")
    mock_view_data_model.return_value = mock_view_training_data_response

    # Act
    df = emulator.view_train_data()

    # Assert
    dataframe_regression.check(df, basename="view_training_data")


@patch("twinlab.api.view_data_model")
def test_view_test_data(
    mock_view_data_model, mock_view_test_data_reponse, dataframe_regression
):
    # Arange
    emulator = tl.Emulator(id="test_emulator")
    mock_view_data_model.return_value = mock_view_test_data_reponse

    # Act
    df = emulator.view_test_data()

    # Assert
    dataframe_regression.check(df, basename="view_test_data")
