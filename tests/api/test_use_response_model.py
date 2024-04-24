# Standard imports
import io

# Third-party imports
import pandas as pd

# Project imports
import twinlab.api as api


def wait_for_use_response_model(method, process_id):
    status = 202
    while status == 202:
        status, body = api.use_response_model(
            model_id="test_model",
            method=method,
            process_id=process_id,
        )
    return status, body


def test_predict(training_setup, dataframe_regression, predict_request):
    # NOTE: training_setup is a necessary argument, even though not used, in order to run the fixture
    status, body = wait_for_use_response_model("predict", predict_request)
    df = pd.read_csv(io.StringIO(body["dataframe"]))
    assert status == 200
    dataframe_regression.check(df)


def test_sample(training_setup, dataframe_regression, sample_request):
    # NOTE: training_setup is a necessary argument, even though not used, in order to run the fixture
    status, body = wait_for_use_response_model("sample", sample_request)
    df = pd.read_csv(io.StringIO(body["dataframe"]))
    assert status == 200
    dataframe_regression.check(df)


def test_get_candidate_points(
    training_setup, dataframe_regression, get_candidate_points_request
):
    # NOTE: training_setup is a necessary argument, even though not used, in order to run the fixture
    status, body = wait_for_use_response_model(
        "get_candidate_points", get_candidate_points_request
    )
    df = pd.read_csv(io.StringIO(body["dataframe"]))
    assert status == 200
    dataframe_regression.check(df)


def test_solve_inverse(training_setup, dataframe_regression, solve_inverse_request):
    # NOTE: training_setup is a necessary argument, even though not used, in order to run the fixture
    status, body = wait_for_use_response_model("solve_inverse", solve_inverse_request)
    df = pd.read_csv(io.StringIO(body["dataframe"]))
    assert status == 200
    dataframe_regression.check(df)
