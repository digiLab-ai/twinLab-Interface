import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import twinlab as tl


class TestEmulator(unittest.TestCase):

    @patch("twinlab.api.summarise_model")
    @patch.object(tl.Emulator, "predict")
    def test_plot(self, mock_predict, mock_summarise_model):

        # Arange
        emulator = tl.Emulator(id="test_emulator")
        x_fixed = {
            "Water [times/week]": 3.0,
        }
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
                "Sunlight [hours/day]": {0: 3.78, 1: 8.28, 2: 12.78},
                "Water [times/week]": {0: 3.0, 1: 3.0, 2: 3.0},
            }
        )
        mock_predict.return_value = pd.DataFrame(
            {"Fruits produced": np.linspace(0, 3, 3)}
        ), pd.DataFrame({"Fruits produced": np.linspace(0, 3, 3)})

        # Act
        fig = emulator.plot(
            "Sunlight [hours/day]", "Fruits produced", x_fixed=x_fixed, n_points=3
        )

        # Extract the dataframe that dataset.predict was called with
        df = mock_predict.call_args[0][0]

        # Assert
        assert isinstance(fig, type(plt))
        pd.testing.assert_frame_equal(expected_df, df)
