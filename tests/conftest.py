import os
import time

import pandas as pd
import pytest
import twinlab as tl


@pytest.fixture(scope="session")
def training_setup():

    resources_path = os.path.join("..", "..", "resources")

    # Parameters for tests
    dataset_id = "biscuits-e2e-test"
    emulator_id = "biscuits-e2e-test"
    inputs = ["Pack price [GBP]", "Number of biscuits per pack"]
    outputs = ["Number of packs sold", "Profit [GBP]"]
    train_test_ratio = 0.8
    seed = 123

    # Load training data
    dataset_path = os.path.join(resources_path, "datasets", "biscuits.csv")
    df = tl.load_dataset(dataset_path)

    # Create and upload dataset
    dataset = tl.Dataset(dataset_id)
    dataset.upload(df)

    # Create and train emulator
    emulator = tl.Emulator(emulator_id)
    params = tl.TrainParams(train_test_ratio=train_test_ratio, seed=seed)
    emulator.train(dataset, inputs, outputs, params=params)

    # Yield results
    yield dataset, emulator

    # Clean up only once after all tests have completed running
    dataset.delete()
    emulator.delete()
