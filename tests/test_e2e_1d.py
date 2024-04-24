import os

import twinlab as tl

# Resources path
# NOTE: This path is relative to where you call the test
# NOTE: Calling this test from python level you do two ".."
# NOTE: Calling this test from inside this file's directory you need four ".."
resources_path = os.path.join("..", "..", "resources")

# Parameters for tests
dataset_id = "quickstart-e2e-test"
emulator_id = "quickstart-e2e-test"
inputs = ["x"]
outputs = ["y"]
train_test_ratio = 0.8
seed = 123
num_points = 1
acq_func = "ExpectedImprovement"

# Load training data
dataset_path = os.path.join(resources_path, "datasets", "quickstart.csv")
df = tl.load_dataset(dataset_path)

# Create and upload dataset
dataset = tl.Dataset(dataset_id)
dataset.upload(df)

# Create and train emulator
emulator = tl.Emulator(emulator_id)
params = tl.TrainParams(train_test_ratio=train_test_ratio, seed=seed)
emulator.train(dataset, inputs, outputs, params=params)


def test_recommend(data_regression, dataframe_regression, num_regression):
    # Call recommend results into dataframe
    recommend_df, acq_func_value = emulator.recommend(
        num_points, acq_func, params=tl.RecommendParams(seed=seed)
    )

    # NOTE: It seems to be necessary to sort the dataframe here in order for the tests to pass
    # NOTE: Data_regression does not have a tolerance argument. Therefore, the dataframe must be rounded before testing.
    # NOTE: We don't know why, but occasionally the order of the dataframe is different
    recommend_df = recommend_df.sort_values(by=["x"])
    recommend_df.reset_index(drop=True, inplace=True)

    # Create a dictionary with indices and column names
    columns_indices = {
        "columns": list(recommend_df.columns),
        "indices": list(recommend_df.index),
    }

    # Check active learning results
    # NOTE: Tolerance is set because occasionally the results are sometimes not exactly the same
    # NOTE: This is the same reason as seems to be necessary for the above sorting
    data_regression.check(columns_indices)
    dataframe_regression.check(recommend_df, default_tolerance=dict(rtol=1e-3))
    num_regression.check(
        {"acq_func_value": acq_func_value},
        default_tolerance=dict(rtol=1e-3),
        basename="test_recommend_acq_func",
    )
