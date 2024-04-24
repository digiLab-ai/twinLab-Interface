from unittest.mock import patch

import pytest
import twinlab as tl


@pytest.fixture
def mock_dataset_response():
    return 200, {
        "dataset": "x,y\n0.6964691855978616,-0.8173739564129022\n0.2861393349503794,0.8876561174050408\n0.2268514535642031,0.921552660721474\n0.5513147690828912,-0.3263338765412979\n0.7194689697855631,-0.8325176123242133\n0.4231064601244609,0.4006686354731812\n0.9807641983846156,-0.1649662650236807\n0.6848297385848633,-0.9607643657025954\n0.4809319014843609,0.3401149876855609\n0.3921175181941505,0.8457949914442409\n"
    }


@patch("twinlab.api.view_dataset")
def test_view(mock_view_dataset, mock_dataset_response, dataframe_regression):
    # Arrange
    dataset = tl.Dataset("dataset")
    mock_view_dataset.return_value = mock_dataset_response

    # Act
    df = dataset.view()

    # Assert
    dataframe_regression.check(df, basename="view_dataset")
