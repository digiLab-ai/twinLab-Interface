from unittest.mock import patch

import pytest
import twinlab as tl


@pytest.fixture
def mock_use_model_score_response():
    return 200, {"dataframe": "Fruits produced\n1.7773277414032562\n"}


@patch("twinlab.api.use_model")
def test_score(
    mock_use_model,
    mock_use_model_score_response,
    dataframe_regression,
):

    # Arange
    emulator = tl.Emulator(id="test_emulator")
    mock_use_model.return_value = mock_use_model_score_response

    # Act
    df = emulator.score()

    # Assert
    dataframe_regression.check(df)
