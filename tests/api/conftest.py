import json
import time

import pandas as pd
import pytest
import twinlab.api as api
import twinlab.utils as utils

dataset_id = "api-test"
model_id = "api-test"


@pytest.fixture
def upload_setup():
    df = pd.DataFrame({"X": [1, 2, 3], "y": [4, 5, 6]})
    api.upload_dataset(dataset_id, df.to_csv(index=False))


@pytest.fixture
def upload_with_url_setup():
    df = pd.DataFrame({"X": [1, 2, 3], "y": [4, 5, 6]})
    _, url = api.generate_upload_url(dataset_id)
    utils.upload_dataframe_to_presigned_url(df, url["url"])


@pytest.fixture
def training_setup(upload_setup):
    # NOTE: upload_setup is a necessary argument, even though not used, in order to run the fixture
    params = {
        "dataset_id": dataset_id,
        "inputs": ["X"],
        "outputs": ["y"],
        "train_test_ratio": 0.8,
    }
    params = json.dumps(params)
    status, body = api.train_request_model(model_id, params, "cpu")
    process_id = body["process_id"]
    status, _ = api.train_response_model(model_id, process_id)
    while status == 202:  # Wait for the model to finish training
        time.sleep(1)
        status, _ = api.train_response_model(model_id, process_id)


@pytest.fixture
def data_csv():
    df = pd.DataFrame({"X": [3, 5, 7]})
    return df.to_csv(index=False)


@pytest.fixture
def inverse_csv():
    df = pd.DataFrame({"y": [12]})
    return df.to_csv(index=False)


@pytest.fixture
def inverse_std_csv():
    df = pd.DataFrame({"y": [0.5]})
    return df.to_csv(index=False)


@pytest.fixture
def predict_request(training_setup, data_csv):
    # NOTE: training_setup is a necessary argument, even though not used, in order to run the fixture
    _, body = api.use_request_model(
        model_id=model_id,
        method="predict",
        data_csv=data_csv,
    )
    return body["process_id"]


@pytest.fixture
def sample_request(training_setup, data_csv):
    # NOTE: training_setup is a necessary argument, even though not used, in order to run the fixture
    _, body = api.use_request_model(
        model_id=model_id,
        method="sample",
        data_csv=data_csv,
        num_samples=3,
        kwargs={"seed": 0},
    )
    return body["process_id"]


@pytest.fixture
def get_candidate_points_request(training_setup):
    # NOTE: training_setup is a necessary argument, even though not used, in order to run the fixture
    _, body = api.use_request_model(
        model_id=model_id,
        method="get_candidate_points",
        acq_func="qNIPV",
        num_points=1,
        kwargs={"seed": 123},
    )
    return body["process_id"]


@pytest.fixture
def solve_inverse_request(training_setup, inverse_csv, inverse_std_csv):
    # NOTE: training_setup is a necessary argument, even though not used, in order to run the fixture
    _, body = api.use_request_model(
        model_id=model_id,
        method="solve_inverse",
        data_csv=inverse_csv,
        data_std_csv=inverse_std_csv,
        kwargs={
            "iterations": 10,
            "force_sequential": "true",
            "seed": 123,
        },
    )
    return body["process_id"]


@pytest.fixture(scope="session", autouse=True)
def cleanup():
    api.delete_dataset(dataset_id)
    api.delete_model(model_id)
