import os
import random

import numpy as np
import pandas as pd
import pytest
import twinlab as tl

# Seed
seed = 123

# Resources path
resources_path = os.path.join("..", "..", "resources")
dataset_path = os.path.join(resources_path, "datasets", "biscuits.csv")
eval_path = os.path.join(resources_path, "campaigns", "biscuits", "eval.csv")

# Parameters for tests
inputs = ["Pack price [GBP]", "Number of biscuits per pack"]
outputs = ["Number of packs sold", "Profit [GBP]"]
train_test_ratio = 0.8
# input_explained_variance = 0.95
# output_explained_variance = 0.95
input_retained_dimensions = 1
output_retained_dimensions = 1

# Seed random number generator
random.seed(seed)
np.random.seed(seed)

# Tolerances
tolerance = dict(rtol=1e-4)  # For dataframe_regression
ndigits = 7  # For data_regression after conversion to numerical values


# Function to convert a dictionary into a dictionary numerical values
def get_numerical_dictionary(summary: dict, ndigits: int) -> dict:
    return {k: round(v, ndigits) for k, v in summary.items() if type(v) in [int, float]}


def test_heteroskedastic(data_regression, dataframe_regression):

    # Parameters
    test_base = "test_heteroskedastic"

    # Upload training data
    df = tl.load_dataset(dataset_path)
    dataset = tl.Dataset(id="biscuits")
    dataset.upload(df)

    # Upload standard deviation data
    # TODO: Using a random-number generator to create test data is a recipe for disaster
    dataset_std = tl.Dataset(id="biscuits_std")
    df_std = pd.DataFrame(
        {
            "Pack price [GBP]": [random.uniform(1.6, 3) for _ in range(12)],
            "Number of biscuits per pack": [random.randint(15, 20) for _ in range(12)],
            "Number of packs sold": [random.randint(2900, 3500) for _ in range(12)],
            "Profit [GBP]": [random.uniform(7000, 17500) for _ in range(12)],
        }
    )
    dataset_std.upload(df_std)

    # Create and train emulator
    estimator_params = tl.EstimatorParams(estimator_type="heteroskedastic_gp")
    params = tl.TrainParams(
        train_test_ratio=train_test_ratio,
        dataset_std=dataset_std,
        estimator_params=estimator_params,
        seed=seed,
    )
    emulator = tl.Emulator(id="biscuits-heteroskedastic")
    emulator.train(dataset, inputs, outputs, params=params)

    # Test view
    view = emulator.view()
    del view["modal_handle"]  # Remove modal handle because it changes every time
    data_regression.check(view, basename=f"{test_base}_view")

    # Test summarise
    summary_dictionary = emulator.summarise()
    estimator_diagnostics = summary_dictionary["estimator_diagnostics"]
    summary_keys = list(estimator_diagnostics.keys())
    data_regression.check(summary_keys, basename=f"{test_base}_keys")
    summary_numerical = get_numerical_dictionary(estimator_diagnostics, ndigits=ndigits)
    data_regression.check(summary_numerical, basename=f"{test_base}_summary")

    # Test predict
    df = tl.load_dataset(eval_path)
    df_mean, df_std = emulator.predict(df)
    dataframe_regression.check(
        df_mean, basename=f"{test_base}_predict_mean", default_tolerance=tolerance
    )
    dataframe_regression.check(
        df_std, basename=f"{test_base}_predict_std", default_tolerance=tolerance
    )

    # Clean up
    emulator.delete()
    dataset.delete()
    dataset_std.delete()


def test_heteroskedastic_with_covar_module():
    # Raise error when covar_module is set for heteroskedastic_gp
    # Note that the hetereoskedastic_gp does not support a covar_module
    with pytest.raises(ValueError) as exc_info:
        tl.EstimatorParams(estimator_type="heteroskedastic_gp", covar_module="RBF")
    assert "The parameter covar_module cannot be set for the estimator" in str(
        exc_info.value
    )


def test_multifidelity(data_regression, dataframe_regression):

    # Parameters
    test_base = "test_multifidelity"

    # Upload training data
    # TODO: Using a random-number generator to create test data is a recipe for disaster
    df = tl.load_dataset(dataset_path)
    fidelity = {"fidelity": np.random.uniform(0, 1, len(df))}  # Fidelity column
    df = pd.concat([df, pd.DataFrame(fidelity)], axis="columns")
    dataset = tl.Dataset(id="biscuits-multifidelity")
    dataset.upload(df)

    # Create and train emulator
    estimator_params = tl.EstimatorParams(estimator_type="multi_fidelity_gp")
    params = tl.TrainParams(
        train_test_ratio=train_test_ratio,
        estimator_params=estimator_params,
        seed=seed,
        fidelity="fidelity",
    )
    emulator = tl.Emulator(id="biscuits-multifidelity")
    emulator.train(dataset, inputs, outputs, params=params)

    # Test view
    view = emulator.view()
    del view["modal_handle"]  # Remove Modal handle because it changes every time
    data_regression.check(view, basename=f"{test_base}_view")

    # Test summarise
    summary_dictionary = emulator.summarise()
    estimator_diagnostics = summary_dictionary["estimator_diagnostics"]
    summary_keys = list(estimator_diagnostics.keys())
    data_regression.check(summary_keys, basename=f"{test_base}_keys")
    summary_numerical = get_numerical_dictionary(estimator_diagnostics, ndigits=ndigits)
    data_regression.check(summary_numerical, basename=f"{test_base}_summary")

    # Test predict
    df = tl.load_dataset(eval_path)
    df_mean, df_std = emulator.predict(df)
    dataframe_regression.check(
        df_mean, basename=f"{test_base}_predict_mean", default_tolerance=tolerance
    )
    dataframe_regression.check(
        df_std, basename=f"{test_base}_predict_std", default_tolerance=tolerance
    )

    # Clean up
    emulator.delete()
    dataset.delete()


def test_model_selection(data_regression, dataframe_regression):

    # Parameters
    test_base = "test_model_selection"

    # Upload training data
    df = tl.load_dataset(dataset_path)
    dataset = tl.Dataset(id="biscuits")
    dataset.upload(df)

    # Create and train emulator
    model_selection_params = tl.ModelSelectionParams(evaluation_metric="BIC", depth=3)
    params = tl.TrainParams(
        model_selection_params=model_selection_params,
        train_test_ratio=train_test_ratio,
        seed=seed,
    )
    emulator = tl.Emulator(id="biscuits-model-selection")
    emulator.train(dataset, inputs, outputs, params=params)

    # Test view
    view = emulator.view()
    del view["modal_handle"]  # Remove modal handle because it changes every time
    data_regression.check(view, basename=f"{test_base}_view")

    # Test summarise
    summary_dictionary = emulator.summarise()
    estimator_diagnostics = summary_dictionary["estimator_diagnostics"]
    summary_keys = list(estimator_diagnostics.keys())
    data_regression.check(summary_keys, basename=f"{test_base}_keys")
    summary_numerical = get_numerical_dictionary(estimator_diagnostics, ndigits=ndigits)
    data_regression.check(summary_numerical, basename=f"{test_base}_summary")

    # Test predict
    df = tl.load_dataset(eval_path)
    df_mean, df_std = emulator.predict(df)
    dataframe_regression.check(
        df_mean, basename=f"{test_base}_predict_mean", default_tolerance=tolerance
    )
    dataframe_regression.check(
        df_std, basename=f"{test_base}_predict_std", default_tolerance=tolerance
    )

    # Clean up
    emulator.delete()
    dataset.delete()


def test_functional(data_regression, dataframe_regression):

    # Parameters
    test_base = "test_functional"

    # Upload training data
    df = tl.load_dataset(dataset_path)
    dataset = tl.Dataset(id="biscuits")
    dataset.upload(df)

    # Create and train emulator
    params = tl.TrainParams(
        train_test_ratio=train_test_ratio,
        input_retained_dimensions=input_retained_dimensions,
        output_retained_dimensions=output_retained_dimensions,
        seed=seed,
    )
    emulator = tl.Emulator(id="biscuits-functional")
    emulator.train(dataset, inputs, outputs, params=params)

    # Test view
    view = emulator.view()
    del view["modal_handle"]  # Remove modal handle because it changes every time
    data_regression.check(view, basename=f"{test_base}_view")

    # Test summarise
    summary_dictionary = emulator.summarise()
    estimator_diagnostics = summary_dictionary["estimator_diagnostics"][
        "base_estimator_diagnostics"
    ]
    summary_keys = list(estimator_diagnostics.keys())
    data_regression.check(summary_keys, basename=f"{test_base}_keys")
    summary_numerical = get_numerical_dictionary(estimator_diagnostics, ndigits=ndigits)
    data_regression.check(summary_numerical, basename=f"{test_base}_summary")

    # extract the transform diagnostics
    transform_diagnostics = summary_dictionary["transform"]
    assert "input" in transform_diagnostics
    assert "output" in transform_diagnostics

    # Isolate input/output dictionaries and check individually
    summary_numerical_input = get_numerical_dictionary(
        transform_diagnostics["input"][0], ndigits=ndigits
    )
    summmary_numerical_output = get_numerical_dictionary(
        transform_diagnostics["output"][0], ndigits=ndigits
    )
    data_regression.check(
        summary_numerical_input, basename="test_functional_summary_input"
    )
    data_regression.check(
        summmary_numerical_output, basename="test_functional_summary_output"
    )

    # Test predict
    df = tl.load_dataset(eval_path)
    df_mean, df_std = emulator.predict(df)
    dataframe_regression.check(
        df_mean, basename=f"{test_base}_predict_mean", default_tolerance=tolerance
    )
    dataframe_regression.check(
        df_std, basename=f"{test_base}_predict_std", default_tolerance=tolerance
    )

    # Clean up
    emulator.delete()
    dataset.delete()


def test_classic(data_regression, dataframe_regression):

    # Parameters
    test_base = "test_classic"

    # Upload training data
    df = tl.load_dataset(dataset_path)
    dataset = tl.Dataset(id="biscuits")
    dataset.upload(df)

    # Create and train emulator
    params = tl.TrainParams(seed=seed)
    emulator = tl.Emulator(id="biscuits-classic")
    emulator.train(dataset, inputs, outputs, params=params)

    # Test view
    view = emulator.view()
    del view["modal_handle"]  # Remove because it changes every time
    data_regression.check(view, basename=f"{test_base}_view")

    # Test summarise
    summary_dictionary = emulator.summarise()
    estimator_diagnostics = summary_dictionary["estimator_diagnostics"]
    summary_keys = list(estimator_diagnostics.keys())
    data_regression.check(summary_keys, basename=f"{test_base}_keys")
    summary_numerical = get_numerical_dictionary(estimator_diagnostics, ndigits=ndigits)
    data_regression.check(summary_numerical, basename=f"{test_base}_summary")

    # Test predict
    df = tl.load_dataset(eval_path)
    df_mean, df_std = emulator.predict(df)
    dataframe_regression.check(
        df_mean, basename=f"{test_base}_predict_mean", default_tolerance=tolerance
    )
    dataframe_regression.check(
        df_std, basename=f"{test_base}_predict_std", default_tolerance=tolerance
    )

    # Clean up
    emulator.delete()
    dataset.delete()
