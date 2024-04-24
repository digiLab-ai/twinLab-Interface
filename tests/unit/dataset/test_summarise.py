from unittest.mock import patch

import pytest
from twinlab import Dataset


@pytest.fixture
def mock_summarise_dataset_response():
    return 200, {
        "dataset_summary": ",x,y\ncount,10.0,10.0\nmean,0.544199352975335,0.029383131672480845\nstd,0.22935216613691597,0.7481906564998719\nmin,0.2268514535642031,-0.9607643657025954\n25%,0.39986475367672814,-0.6946139364450011\n50%,0.5161233352836261,0.0875743613309401\n75%,0.693559323844612,0.7345134024514759\nmax,0.9807641983846156,0.921552660721474\n"
    }


@patch("twinlab.api.summarise_dataset")
def test_summarise(
    mock_summarise_dataset, mock_summarise_dataset_response, dataframe_regression
):
    # Arrange
    dataset = Dataset("dataset")
    mock_summarise_dataset.return_value = mock_summarise_dataset_response

    # Act
    df = dataset.summarise()

    # Assert
    dataframe_regression.check(df, basename="summarise_dataset")
