import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import twinlab as tl


@pytest.fixture
def summarise_model_response():
    return 200, {
        "model_summary": {
            "data_diagnostics": {
                "inputs": {
                    "Sunlight [hours/day]": {
                        "min": 3.78,
                        "max": 12.78,
                        "mean": 8.78,
                    },
                    "Water [times/week]": {
                        "min": 1.12,
                        "max": 5.12,
                        "mean": 3.12,
                    },
                },
                "outputs": {
                    "Fruits produced": {
                        "min": 0.44,
                        "max": 2.44,
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


class TestEmulator(unittest.TestCase):

    @patch("twinlab.api.summarise_model")
    @patch.object(tl.Emulator, "predict")
    def test_plot(self, mock_predict, mock_summarise_model):

        # Arange
        emulator = tl.Emulator(id="test_emulator")
        mock_summarise_model.return_value = 200, {
            "model_summary": {
                "data_diagnostics": {
                    "inputs": {
                        "Sunlight [hours/day]": {
                            "min": 3.78,
                            "max": 12.78,
                            "mean": 8.78,
                        },
                        "Water [times/week]": {
                            "min": 1.12,
                            "max": 5.12,
                            "mean": 3.12,
                        },
                    },
                    "outputs": {
                        "Fruits produced": {
                            "min": 0.44,
                            "max": 2.44,
                            "mean": 1.44,
                        }
                    },
                },
            }
        }
        expected_df = pd.DataFrame(
            {
                "Sunlight [hours/day]": {
                    0: 3.78,
                    1: 8.28,
                    2: 12.78,
                    3: 3.78,
                    4: 8.28,
                    5: 12.78,
                    6: 3.78,
                    7: 8.28,
                    8: 12.78,
                },
                "Water [times/week]": {
                    0: 1.12,
                    1: 1.12,
                    2: 1.12,
                    3: 3.12,
                    4: 3.12,
                    5: 3.12,
                    6: 5.12,
                    7: 5.12,
                    8: 5.12,
                },
            }
        )
        mock_predict.return_value = pd.DataFrame(
            {"Fruits produced": np.linspace(0, 3**2, 3**2)}
        ), pd.DataFrame({"Fruits produced": np.linspace(0, 3**2, 3**2)})

        # Act
        fig = emulator.heatmap(
            "Sunlight [hours/day]", "Water [times/week]", "Fruits produced", n_points=3
        )

        # Extract the dataframe that dataset.predict was called with
        df = mock_predict.call_args[0][0]

        # Assert
        assert isinstance(fig, type(plt))
        pd.testing.assert_frame_equal(expected_df, df)
