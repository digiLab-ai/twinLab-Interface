from unittest.mock import patch

import pytest
import twinlab as tl


@pytest.fixture
def priors():
    return [
        tl.Prior("x1", tl.distributions.Uniform(0, 12)),
        tl.Prior("x2", tl.distributions.Uniform(0, 0.5)),
        tl.Prior("x3", tl.distributions.Uniform(0, 10)),
    ]


@pytest.fixture
def mock_response():
    return 200, {
        "initial_design": "x1,x2\n0.8333333333333334,0.8333333333333334\n0.16666666666666666,0.5\n0.5,0.16666666666666666\n"
    }


@patch("twinlab.api.get_initial_design")
def test_design(mock_get_initial_design, priors, mock_response, dataframe_regression):

    # Arange
    emulator = tl.Emulator(id="test_emulator")
    mock_get_initial_design.return_value = mock_response

    # Act
    df = emulator.design(priors, 3)

    # Assert
    dataframe_regression.check(df, basename="initial_design")
