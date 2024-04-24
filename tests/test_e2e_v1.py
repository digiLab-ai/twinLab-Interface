import os

import twinlab as tl
import pandas as pd

# Resources path
# NOTE: This path is relative to where you call the test
# NOTE: Calling this test from python level you do two ".."
# NOTE: Calling this test from inside this file's directory you need four ".."
resources_path = os.path.join("..", "..", "resources")

# Campaign parameters
campaign_id = "biscuits"
campaign_path = os.path.join(resources_path, "campaigns", "biscuits", "params.json")

# Training dataset path
dataset_id = "biscuits"
dataset_path = os.path.join(resources_path, "datasets", "biscuits.csv")

# Test dataset path
predict_path = os.path.join(resources_path, "campaigns", "biscuits", "eval.csv")

# Observation path for solving Bayesian inverse problem
obs_path = os.path.join(resources_path, "campaigns", "biscuits", "obs.csv")
obs_std_path = os.path.join(resources_path, "campaigns", "biscuits", "obs_std.csv")

verbose = False
debug = False
num_samples = 5
num_points = 5

seed = 123

# Set up the pipeline for the prediction test
tl.upload_dataset(dataset_path, dataset_id, verbose=verbose, debug=debug)
tl.train_campaign(campaign_path, campaign_id, verbose=verbose, debug=debug)


def test_score_campaign(data_regression, dataframe_regression):
    # Call the score campaign into dataframe
    df = tl.score_campaign(
        campaign_id,
        combined_score=False,
        verbose=verbose,
        debug=debug,
    )

    # Create a dictionary with indices and column names
    columns_indices = {
        "columns": list(df.columns),
        "indices": list(df.index),
    }

    # Check the score campaign results
    data_regression.check(columns_indices)
    dataframe_regression.check(df)


def test_get_calibration_curve_campaign(data_regression, dataframe_regression):
    # Call the get_calibration_curve campaign into dataframe
    df = tl.get_calibration_curve_campaign(
        campaign_id,
        type="quantile",
        verbose=verbose,
        debug=debug,
    )

    # Create a dictionary with indices and column names
    columns_indices = {
        "columns": list(df.columns),
        "indices": list(df.index),
    }

    # Check the score campaign results
    data_regression.check(columns_indices)
    dataframe_regression.check(df)


def test_predict_pipeline(data_regression, dataframe_regression):
    # Call predict campaign and retrieve predicted mean, std
    predict_df_mean, predict_df_std = tl.predict_campaign(
        predict_path, campaign_id, verbose=verbose, debug=debug
    )

    # Create dataframe from predict with mean, std
    predict_df = pd.concat([predict_df_mean, predict_df_std])

    # Create a dictionary with indices and column names
    columns_indices = {
        "columns": list(predict_df.columns),
        "indices": list(predict_df.index),
    }

    # Check predict results
    data_regression.check(columns_indices)
    dataframe_regression.check(predict_df)


def test_sample_pipeline(data_regression, dataframe_regression):
    # Call sample results into dataframe
    sample_df = tl.sample_campaign(
        predict_path,
        campaign_id,
        num_samples,
        verbose=verbose,
        debug=debug,
        kwargs={"seed": seed},
    )

    # Create a dictionary with indices and column names
    columns_indices = {
        "columns": list(sample_df.columns),
        "indices": list(sample_df.index),
    }

    # Check active learning results
    data_regression.check(columns_indices)
    dataframe_regression.check(sample_df)


def test_active_pipeline(data_regression, dataframe_regression):
    # Call active learning results into dataframe
    active_df = tl.active_learn_campaign(
        campaign_id,
        num_points,
        verbose=verbose,
        debug=debug,
        kwargs={"seed": seed},
    )

    # NOTE: It seems to be necessary to sort the dataframe here but we dont know why
    active_df = active_df.sort_values(by=["Pack price [GBP]"])
    active_df.reset_index(drop=True, inplace=True)

    # Create a dictionary with indices and column names
    columns_indices = {
        "columns": list(active_df.columns),
        "indices": list(active_df.index),
    }

    # Check active learning results
    # NOTE: Tolerance is set because the results are sometimes not exactly the same
    data_regression.check(columns_indices)
    dataframe_regression.check(active_df, default_tolerance=dict(rtol=1e-3))


def test_optimisation_pipeline(data_regression, dataframe_regression):
    # Call optimise_campaign campaign into dataframe
    maxima_candidates = tl.optimise_campaign(
        campaign_id,
        num_points,
        verbose=verbose,
        debug=debug,
        kwargs={"seed": seed},
    )

    # NOTE: It seems to be necessary to sort the dataframe here but we dont know why
    maxima_candidates = maxima_candidates.sort_values(by=["Pack price [GBP]"])
    maxima_candidates.reset_index(drop=True, inplace=True)

    # Create a dictionary with indices and column names
    columns_indices = {
        "columns": list(maxima_candidates.columns),
        "indices": list(maxima_candidates.index),
    }

    # Check optimise_campaign results
    # NOTE: Tolerance is set because the results are sometimes not exactly the same
    data_regression.check(columns_indices)
    dataframe_regression.check(maxima_candidates, default_tolerance=dict(rtol=1e-3))


def test_inverse_pipeline(data_regression, dataframe_regression):
    # Call inverse methods campaign into dataframe
    inverse = tl.solve_inverse_campaign(
        campaign_id,
        obs_path,
        obs_std_path,
        verbose=verbose,
        debug=debug,
        kwargs={
            "force_sequential": False,
            "n_chains": 1,
            "iterations": 100,
            "seed": seed,
        },
    )

    # Create a dictionary with indices and column names
    columns_indices = {
        "columns": list(inverse.columns),
        "indices": list(inverse.index),
    }

    # Check inverse methods results
    data_regression.check(columns_indices)
    dataframe_regression.check(inverse)
