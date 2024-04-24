import unittest
from unittest.mock import patch

import pandas as pd
import pytest
import twinlab as tl


class TestEmulator(unittest.TestCase):

    @patch.object(tl.Emulator, "train")
    @patch.object(tl.Emulator, "recommend")
    @patch.object(tl.Dataset, "upload")
    @patch.object(tl.Emulator, "view_train_data")
    def test_learn(self, mock_view_train_data, mock_upload, mock_recommend, mock_train):

        # Arange
        emulator = tl.Emulator(id="test_emulator")
        dataset = tl.Dataset(id="test_dataset")
        inputs = ["X"]
        outputs = ["y"]

        # Create a mock simulation function that returns a DataFrame
        def simulation(inputs):
            return {
                "y": [2.0, 3.0, 3.0, 4.0, 2.0],
            }

        mock_recommend.return_value = (
            pd.DataFrame({"X": [0.1, 0.2, 0.3], "y": [0.4, 0.5, 0.6]}),
            1.0,
        )

        mock_view_train_data.return_value = pd.DataFrame(
            {"X": [0.1, 0.2, 0.3], "y": [0.4, 0.5, 0.6]}
        )

        expected_df = pd.DataFrame(
            {"X": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3], "y": [0.4, 0.5, 0.6, 2.0, 3.0, 3.0]}
        )

        # Act
        emulator.learn(
            dataset=dataset,
            inputs=inputs,
            outputs=outputs,
            num_loops=3,
            num_points_per_loop=10,
            acq_func="ExpectedImprovement",
            simulation=simulation,
        )

        # Extract the dataframe that dataset.upload was called with
        df = mock_upload.call_args[0][0]

        # Assert
        self.assertTrue(expected_df.equals(df))
        self.assertEqual(mock_train.call_count, 4)
        self.assertEqual(mock_recommend.call_count, 3)

    def test_bad_test_train_ratio_fails(self):

        # Test a variety of bad test_train_ratios to be sure a ValueError is raised each time
        for i in [0.8, 1.1, 10]:
            with pytest.raises(
                ValueError,
                match=f"test_train_ratio must be set to 1, not {i}, for this method to work.",
            ):

                params = tl.TrainParams(train_test_ratio=i)

                emulator = tl.Emulator(id="test_emulator")
                dataset = tl.Dataset(id="test_dataset")
                inputs = ["X"]
                outputs = ["y"]

                # Create a mock simulation function that returns a DataFrame
                def simulation(inputs):
                    return {
                        "y": [2.0, 3.0, 3.0, 4.0, 2.0],
                    }

                emulator.learn(
                    dataset=dataset,
                    inputs=inputs,
                    outputs=outputs,
                    num_loops=3,
                    num_points_per_loop=10,
                    acq_func="ExpectedImprovement",
                    simulation=simulation,
                    train_params=params,
                )

    def test_bad_num_loops_fails(self):

        # Test to be sure a ValueError is raised each time when num_loops = 0
        with pytest.raises(
            ValueError,
            match=f"num_loops must be set to an integer value of 1 or more, not 0, for this method to work.",
        ):

            params = tl.TrainParams(train_test_ratio=1)

            emulator = tl.Emulator(id="test_emulator")
            dataset = tl.Dataset(id="test_dataset")
            inputs = ["X"]
            outputs = ["y"]

            # Create a mock simulation function that returns a DataFrame
            def simulation(inputs):
                return {
                    "y": [2.0, 3.0, 3.0, 4.0, 2.0],
                }

            emulator.learn(
                dataset=dataset,
                inputs=inputs,
                outputs=outputs,
                num_loops=0,
                num_points_per_loop=10,
                acq_func="ExpectedImprovement",
                simulation=simulation,
                train_params=params,
            )
