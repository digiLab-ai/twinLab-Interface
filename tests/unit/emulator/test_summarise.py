from unittest.mock import patch

import pytest
import twinlab as tl


@pytest.fixture
def mock_summarise_model_response():
    return 200, {
        "model_summary": {
            "data_diagnostics": {
                "inputs": {
                    "Sunlight [hours/day]": {
                        "mean": 8.78,
                    },
                    "Water [times/week]": {
                        "mean": 3.12,
                    },
                },
                "outputs": {
                    "Fruits produced": {
                        "mean": 1.44,
                    }
                },
            },
            "estimator_diagnostics": {
                "covar_module": "RBFKernel",
            },
            "transformer_diagnostics": [],
        }
    }


@pytest.fixture
def mock_summarise_model_return_value():
    return {
        "estimator_diagnostics": {
            "covar_module": "RBFKernel",
        },
        "transformer_diagnostics": [],
    }


@patch("twinlab.api.summarise_model")
def test_summarise(
    mock_summarise_model,
    mock_summarise_model_response,
    mock_summarise_model_return_value,
):

    # Arange
    emulator = tl.Emulator(id="test_emulator")
    mock_summarise_model.return_value = mock_summarise_model_response

    # Act
    summary = emulator.summarise()

    # Assert
    assert summary == mock_summarise_model_return_value
