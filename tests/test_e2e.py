import os

import pandas as pd
import pytest
import twinlab as tl

# Resources path
# NOTE: This path is relative to where you call the test
# NOTE: Calling this test from python level you do two ".."
# NOTE: Calling this test from inside this file's directory you need four ".."
resources_path = os.path.join("..", "..", "resources")

inputs = ["Pack price [GBP]", "Number of biscuits per pack"]
outputs = ["Number of packs sold", "Profit [GBP]"]
# train_test_ratio = 0.8
seed = 123
num_samples = 5
num_samples_large = int(
    4e4
)  # This ensures that the dataframe returned is larger than 6 MB
num_points = 5
priors = [
    tl.Prior("x1", tl.distributions.Uniform(0, 12)),
    tl.Prior("x2", tl.distributions.Uniform(0, 0.5)),
    tl.Prior("x3", tl.distributions.Uniform(0, 10)),
]

# Load data for prediction and sample
eval_path = os.path.join(resources_path, "campaigns", "biscuits", "eval.csv")
eval_df = tl.load_dataset(eval_path)

# Load data for calibration
obs_path = os.path.join(resources_path, "campaigns", "biscuits", "obs.csv")
obs_std_path = os.path.join(resources_path, "campaigns", "biscuits", "obs_std.csv")
obs_df = tl.load_dataset(obs_path)
obs_std_df = tl.load_dataset(obs_std_path)


# @pytest.mark.usefixtures("cleanup")
def test_analyse_dataset(training_setup, dataframe_regression):
    dataset, _ = training_setup
    variance_df = dataset.analyse_variance(columns=inputs)
    # Check svd is correct
    dataframe_regression.check(variance_df)


def test_design(training_setup, data_regression, dataframe_regression):
    # Call the design method
    _, emulator = training_setup

    initial_design = emulator.design(priors, 10, params=tl.DesignParams(seed=seed))

    # Create a dictionary with indices and column names
    columns_indices = {
        "columns": list(initial_design.columns),
        "indices": list(initial_design.index),
    }

    # Check the design results
    data_regression.check(columns_indices)
    dataframe_regression.check(initial_design)


def test_score(training_setup, data_regression, dataframe_regression):
    _, emulator = training_setup
    # Call the score emulator into dataframe
    df = emulator.score()

    # Create a dictionary with indices and column names
    columns_indices = {
        "columns": list(df.columns),
        "indices": list(df.index),
    }

    # Check the score emulator results
    data_regression.check(columns_indices)
    dataframe_regression.check(df)


def test_combined_score(training_setup, num_regression):
    _, emulator = training_setup
    # Call the score emulator into dataframe
    score_params = tl.ScoreParams(combined_score=True)
    final_score = emulator.score(params=score_params)

    # Create a dictionary of the final score
    score_dict = {"combined_score": final_score}

    # Check the score emulator results
    num_regression.check(score_dict, default_tolerance=dict(rtol=1e-4))


def test_benchmark(training_setup, data_regression, dataframe_regression):
    _, emulator = training_setup
    # Call the benchmark emulator method and convert output to dataframe
    df = emulator.benchmark()
    # df = pd.DataFrame(curve, columns=["X", "Y"])

    # Create a dictionary with indices and column names
    columns_indices = {
        "columns": list(df.columns),
        "indices": list(df.index),
    }

    # Check the score campaign results
    data_regression.check(columns_indices)
    dataframe_regression.check(df)


def test_benchmark_with_interval(training_setup, data_regression, dataframe_regression):
    _, emulator = training_setup
    # Call the benchmark emulator method and convert output to dataframe
    benchmark_params = tl.BenchmarkParams(type="interval")
    df = emulator.benchmark(params=benchmark_params)

    # Create a dictionary with indices and column names
    columns_indices = {
        "columns": list(df.columns),
        "indices": list(df.index),
    }

    # Check the score campaign results
    data_regression.check(columns_indices)
    dataframe_regression.check(df)


def test_predict(training_setup, data_regression, dataframe_regression):
    _, emulator = training_setup
    # Running predict method
    predict_df_mean, predict_df_std = emulator.predict(eval_df)

    # Concatenating to a single data frame
    predict_df = pd.concat([predict_df_mean, predict_df_std])

    # Create a dictionary with indices and column names
    columns_indices = {
        "columns": list(predict_df.columns),
        "indices": list(predict_df.index),
    }

    # Checking that it returns the right dataframe
    data_regression.check(columns_indices)
    dataframe_regression.check(predict_df)


def test_predict_without_noise(training_setup, data_regression, dataframe_regression):
    _, emulator = training_setup
    # Running predict method
    predict_params = tl.PredictParams(observation_noise=False)
    predict_df_mean, predict_df_std = emulator.predict(eval_df, params=predict_params)

    # Concatenating to a single data frame
    predict_df = pd.concat([predict_df_mean, predict_df_std])

    # Create a dictionary with indices and column names
    columns_indices = {
        "columns": list(predict_df.columns),
        "indices": list(predict_df.index),
    }

    # Checking that it returns the right dataframe
    data_regression.check(columns_indices)
    dataframe_regression.check(predict_df)


@pytest.mark.timeout(30)  # Timeout for this test given its tendency to hang on fail
def test_predict_with_wrong_input(training_setup):
    eval_df = pd.DataFrame(
        {"not-an-input": [1.0, 2.0, 3.0]}
    )  # NOTE: Emulator has no "not-an-input" input
    _, emulator = training_setup
    with pytest.raises(Exception):  # Ensure that exception is raised
        emulator.predict(eval_df)


def test_sample(training_setup, data_regression, dataframe_regression):
    _, emulator = training_setup
    # Call sample results into dataframe
    sample_df = emulator.sample(eval_df, num_samples, params=tl.SampleParams(seed=seed))

    # Create a dictionary with indices and column names
    columns_indices = {
        "columns": list(sample_df.columns),
        "indices": list(sample_df.index),
    }

    data_regression.check(columns_indices)
    dataframe_regression.check(sample_df)


def test_sample_large(training_setup, data_regression, dataframe_regression):

    _, emulator = training_setup

    # Call sample results into dataframe
    sample_df = emulator.sample(
        eval_df, num_samples_large, params=tl.SampleParams(seed=seed)
    )

    # Create a dictionary with indices and column names
    columns_indices = {
        "columns": list(sample_df.columns),
        "indices": list(sample_df.index),
    }

    data_regression.check(columns_indices)
    dataframe_regression.check(sample_df)


def test_recommend_optimise(
    training_setup, data_regression, dataframe_regression, num_regression
):
    _, emulator = training_setup
    # Call recommend results into dataframe
    recommend_df, acq_func_value = emulator.recommend(
        num_points,
        acq_func="qExpectedImprovement",
        params=tl.RecommendParams(seed=seed),
    )

    # NOTE: It seems to be necessary to sort the dataframe here in order for the tests to pass
    # NOTE: We don't know why, but occasionally the order of the dataframe is different
    recommend_df = recommend_df.sort_values(by=["Pack price [GBP]"])
    recommend_df.reset_index(drop=True, inplace=True)

    # Create a dictionary with indices and column names
    columns_indices = {
        "columns": list(recommend_df.columns),
        "indices": list(recommend_df.index),
    }

    # Check active learning results
    # NOTE: Tolerance is set because occasionally the results are sometimes not exactly the same
    data_regression.check(columns_indices)
    dataframe_regression.check(recommend_df, default_tolerance=dict(rtol=1e-3))
    num_regression.check(
        {"acq_func_value": acq_func_value},
        default_tolerance=dict(rtol=1e-3),
        basename="test_recommend_optimise_acq_func",
    )


def test_recommend_active_learn(
    training_setup, data_regression, dataframe_regression, num_regression
):
    _, emulator = training_setup
    # Call recommend results into dataframe
    recommend_df, acq_func_value = emulator.recommend(
        num_points,
        acq_func="qNegIntegratedPosteriorVariance",
        params=tl.RecommendParams(seed=seed),
    )

    # NOTE: It seems to be necessary to sort the dataframe here in order for the tests to pass
    # NOTE: We don't know why, but occasionally the order of the dataframe is different
    recommend_df = recommend_df.sort_values(by=["Pack price [GBP]"])
    recommend_df.reset_index(drop=True, inplace=True)

    # Create a dictionary with indices and column
    columns_indices = {
        "columns": list(recommend_df.columns),
        "indices": list(recommend_df.index),
    }

    # Check active learning results
    # NOTE: Tolerance is set because occasionally the results are sometimes not exactly the same
    data_regression.check(columns_indices)
    dataframe_regression.check(recommend_df, default_tolerance=dict(rtol=1e-3))
    num_regression.check(
        {"acq_func_value": acq_func_value},
        default_tolerance=dict(rtol=1e-3),
        basename="test_recommend_active_learn_acq_func",
    )


def test_calibrate(training_setup, data_regression, dataframe_regression):
    _, emulator = training_setup
    # Call calibrate method
    params = tl.CalibrateParams(
        n_chains=1,
        iterations=100,
        seed=seed,
    )
    calibrate_df = emulator.calibrate(obs_df, obs_std_df, params)

    # Create a dictionary with indices and column names
    columns_indices = {
        "columns": list(calibrate_df.columns),
        "indices": list(calibrate_df.index),
    }

    # Check inverse methods results
    data_regression.check(columns_indices)
    dataframe_regression.check(calibrate_df)
