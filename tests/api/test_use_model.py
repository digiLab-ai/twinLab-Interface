# Standard imports
import io

# Third-party imports
import pandas as pd

# Project imports
import twinlab.api as api


def test_score(training_setup, dataframe_regression):
    # NOTE: training_setup is a necessary argument, even though not used, in order to run the fixture
    status, body = api.use_model(
        model_id="test_model",
        method="score",
        metric="MSE",
    )
    df = pd.read_csv(io.StringIO(body["dataframe"]))
    assert status == 200
    dataframe_regression.check(df)


def test_get_calibration_curve(training_setup, dataframe_regression):
    # NOTE: training_setup is a necessary argument, even though not used, in order to run the fixture
    status, body = api.use_model(
        model_id="test_model",
        method="get_calibration_curve",
        type="quantile",
    )
    df = pd.read_csv(io.StringIO(body["dataframe"]))
    assert status == 200
    dataframe_regression.check(df)


def test_predict(training_setup, data_csv, dataframe_regression):
    # NOTE: training_setup is a necessary argument, even though not used, in order to run the fixture
    status, body = api.use_model(
        model_id="test_model",
        method="predict",
        data_csv=data_csv,
        kwargs={"seed": 123},
    )
    df = pd.read_csv(io.StringIO(body["dataframe"]))
    assert status == 200
    dataframe_regression.check(df)


def test_sample(training_setup, data_csv, dataframe_regression):
    # NOTE: training_setup is a necessary argument, even though not used, in order to run the fixture
    status, body = api.use_model(
        model_id="test_model",
        method="sample",
        data_csv=data_csv,
        num_samples=3,
        kwargs={"seed": 123},
    )
    df = pd.read_csv(io.StringIO(body["dataframe"]))
    assert status == 200
    dataframe_regression.check(df)


def test_get_candidate_points(training_setup, dataframe_regression):
    # NOTE: training_setup is a necessary argument, even though not used, in order to run the fixture
    status, body = api.use_model(
        model_id="test_model",
        method="get_candidate_points",
        acq_func="qNIPV",
        num_points=1,
        kwargs={"seed": 123},
    )
    df = pd.read_csv(io.StringIO(body["dataframe"]))
    assert status == 200
    dataframe_regression.check(df)


def test_solve_inverse(
    training_setup, inverse_csv, inverse_std_csv, dataframe_regression
):
    # NOTE: training_setup is a necessary argument, even though not used, in order to run the fixture
    status, body = api.use_model(
        model_id="test_model",
        method="solve_inverse",
        data_csv=inverse_csv,
        data_std_csv=inverse_std_csv,
        kwargs={
            "iterations": 10,
            "force_sequential": "true",
            "seed": 123,
        },
    )
    df = pd.read_csv(io.StringIO(body["dataframe"]))
    assert status == 200
    dataframe_regression.check(df)
