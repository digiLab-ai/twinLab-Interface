from unittest.mock import patch

import pytest
import twinlab as tl


@pytest.fixture
def mock_use_model_benchmark_response():
    return 200, {
        "dataframe": "Fruits produced\n0.0\n0.0\n0.0\n0.0\n0.0\n0.0\n0.0\n0.0\n"
    }


@patch("twinlab.api.use_model")
def test_benchmark(
    mock_use_model,
    mock_use_model_benchmark_response,
    dataframe_regression,
):

    # Arange
    emulator = tl.Emulator(id="test_emulator")
    mock_use_model.return_value = mock_use_model_benchmark_response

    # Act
    df = emulator.benchmark()

    # Assert
    dataframe_regression.check(df)
