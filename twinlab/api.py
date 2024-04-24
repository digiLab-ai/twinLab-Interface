# Standard imports
import os
from typing import Dict, Optional, Tuple, Union

# Third-party imports
import requests
from typeguard import typechecked

### Helper functions ###


def _create_headers(verbose: bool = False) -> Dict[str, str]:
    headers = {
        "X-API-Key": os.getenv("TWINLAB_API_KEY"),
        "X-Language": "python",
    }
    verbose_str = "true" if verbose else "false"
    headers["X-Verbose"] = verbose_str
    return headers


def _get_response_body(response: requests.Response) -> Union[dict, str]:
    # TODO: Use attribute of response to check if json/text
    try:
        return response.json()
    except:
        return response.text


def check_status_code(func):
    def wrapper(*args, **kwargs):
        status, body = func(*args, **kwargs)
        if not str(status).startswith("2"):
            print("Non-success status code")
            raise Exception(f"Error Code: {status} - {body}")
        return status, body

    return wrapper


###  ###

### API ###


@typechecked
@check_status_code
def get_user(verbose: Optional[bool] = False) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/user"
    headers = _create_headers(verbose=verbose)
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = _get_response_body(response)
    return status, body


@typechecked
@check_status_code
def get_versions(verbose: Optional[bool] = False) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/versions"
    headers = _create_headers(verbose=verbose)
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = _get_response_body(response)
    return status, body


@typechecked
@check_status_code
def generate_upload_url(
    dataset_id: str, verbose: Optional[bool] = False
) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/upload_url/{dataset_id}"
    headers = _create_headers(verbose=verbose)
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = _get_response_body(response)
    return status, body


@typechecked
@check_status_code
def generate_temp_upload_url(
    dataset_id: str, verbose: Optional[bool] = False
) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/temp_upload_url/{dataset_id}"
    headers = _create_headers(verbose=verbose)
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = _get_response_body(response)
    return status, body


@typechecked
@check_status_code
def process_uploaded_dataset(
    dataset_id: str, verbose: Optional[bool] = False
) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/datasets/{dataset_id}"
    headers = _create_headers(verbose=verbose)
    response = requests.post(url, headers=headers)
    status = response.status_code
    body = _get_response_body(response)
    return status, body


@typechecked
@check_status_code
def upload_dataset(
    dataset_id: str, data_csv: str, verbose: bool = False
) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/datasets/{dataset_id}"
    headers = _create_headers(verbose=verbose)
    request_body = {"dataset": data_csv}
    response = requests.put(url, headers=headers, json=request_body)
    status = response.status_code
    body = _get_response_body(response)
    return status, body


# NOTE: Columns is currently a list but it might have to be inserted and read as a sting
@typechecked
@check_status_code
def analyse_dataset(
    dataset_id: str, columns: str, verbose: Optional[bool] = False
) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/datasets/{dataset_id}/analysis"
    headers = _create_headers(verbose=verbose)
    query_params = {"columns": columns}
    response = requests.get(url, headers=headers, params=query_params)
    status = response.status_code
    body = _get_response_body(response)
    return status, body


@typechecked
@check_status_code
def list_datasets(verbose: Optional[bool] = False) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/datasets"
    headers = _create_headers(verbose=verbose)
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = _get_response_body(response)
    return status, body


@typechecked
@check_status_code
def view_dataset(dataset_id: str, verbose: Optional[bool] = False) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/datasets/{dataset_id}"
    headers = _create_headers(verbose=verbose)
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = _get_response_body(response)
    return status, body


@typechecked
@check_status_code
def load_example_dataset(dataset_id: str, verbose: Optional[bool] = False) -> dict:
    url = f"{os.getenv('TWINLAB_URL')}/example_datasets/{dataset_id}"
    headers = _create_headers(verbose=verbose)
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = _get_response_body(response)
    return status, body


@typechecked
@check_status_code
def list_example_datasets(verbose: Optional[bool] = False) -> dict:
    url = f"{os.getenv('TWINLAB_URL')}/example_datasets"
    headers = _create_headers(verbose=verbose)
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = _get_response_body(response)
    return status, body


@typechecked
@check_status_code
def summarise_dataset(
    dataset_id: str, verbose: Optional[bool] = False
) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/datasets/{dataset_id}/summarise"
    headers = _create_headers(verbose=verbose)
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = _get_response_body(response)
    return status, body


@typechecked
@check_status_code
def delete_dataset(
    dataset_id: str, verbose: Optional[bool] = False
) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/datasets/{dataset_id}"
    headers = _create_headers(verbose=verbose)
    response = requests.delete(url, headers=headers)
    status = response.status_code
    body = _get_response_body(response)
    return status, body


@typechecked
@check_status_code
def delete_temp_dataset(
    dataset_id: str, verbose: Optional[bool] = False
) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/temp_datasets/{dataset_id}"
    headers = _create_headers(verbose=verbose)
    response = requests.delete(url, headers=headers)
    status = response.status_code
    body = _get_response_body(response)
    return status, body


@typechecked
@check_status_code
def view_data_model(
    model_id: str, dataset_type: str, verbose: Optional[bool] = False
) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/models/{model_id}/view_data_model"
    query_params = {"dataset_type": dataset_type}
    headers = _create_headers(verbose=verbose)
    response = requests.get(url, headers=headers, params=query_params)
    status = response.status_code
    body = _get_response_body(response)
    return status, body


@typechecked
@check_status_code
def list_models(verbose: Optional[bool] = False) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/models"
    headers = _create_headers(verbose=verbose)
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = _get_response_body(response)
    return status, body


@typechecked
@check_status_code
def list_processes_model(
    model_id: str, verbose: Optional[bool] = False
) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/models/{model_id}/processes"
    headers = _create_headers(verbose=verbose)
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = _get_response_body(response)
    return status, body


@typechecked
@check_status_code
def get_status_model(
    model_id: str, verbose: Optional[bool] = False
) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/models/{model_id}"
    headers = _create_headers(verbose=verbose)
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = _get_response_body(response)
    return status, body


@typechecked
@check_status_code
def view_model(model_id: str, verbose: Optional[bool] = False) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/models/{model_id}/view"
    headers = _create_headers(verbose=verbose)
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = _get_response_body(response)
    return status, body


@typechecked
@check_status_code
def summarise_model(model_id: str, verbose: Optional[bool] = False) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/models/{model_id}/summarise"
    headers = _create_headers(verbose=verbose)
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = _get_response_body(response)
    return status, body


@typechecked
@check_status_code
def delete_model(model_id: str, verbose: Optional[bool] = False) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/models/{model_id}"
    headers = _create_headers(verbose=verbose)
    response = requests.delete(url, headers=headers)
    status = response.status_code
    body = _get_response_body(response)
    return status, body


### Training ###


@typechecked
@check_status_code
def train_model(
    model_id: str, parameters_json: str, processor: str, verbose: bool = False
) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/models/{model_id}"
    headers = _create_headers(verbose=verbose)
    headers["X-Processor"] = processor
    request_body = {
        # TODO: Add dataset_id and dataset_std_id as keys?
        # TODO: Split this into setup/train_params as in twinLab?
        "parameters": parameters_json,
    }
    response = requests.put(url, headers=headers, json=request_body)
    status = response.status_code
    body = _get_response_body(response)

    return status, body


@typechecked
@check_status_code
def get_status_model(model_id: str, verbose: bool = False) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/models/{model_id}"
    headers = _create_headers(verbose=verbose)
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = _get_response_body(response)
    return status, body


@typechecked
@check_status_code
def train_request_model(
    model_id: str, parameters_json: str, processor: str, verbose: bool = False
) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/models/{model_id}/async/train"
    headers = _create_headers(verbose=verbose)
    headers["X-Processor"] = processor
    request_body = {
        # TODO: Add dataset_id and dataset_std_id as keys?
        # TODO: Split this into setup/train_params as in twinLab?
        "parameters": parameters_json,
    }
    response = requests.put(url, headers=headers, json=request_body)
    status = response.status_code
    body = _get_response_body(response)
    return status, body


@typechecked
@check_status_code
def train_response_model(
    model_id: str, process_id: str, verbose: bool = False
) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/models/{model_id}/async/train/{process_id}"
    headers = _create_headers(verbose=verbose)
    headers["process_id"] = process_id
    headers["model_id"] = model_id
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = _get_response_body(response)
    return status, body


### Synchronous endpoints ###


def _use_request_body(
    data_csv: Optional[str] = None,
    data_std_csv: Optional[str] = None,
    dataset_id: Optional[str] = None,
    dataset_std_id: Optional[str] = None,
    **kwargs,
) -> dict:
    request_body = {"kwargs": kwargs}
    if data_csv is not None:
        request_body["dataset"] = data_csv
    if data_std_csv is not None:
        request_body["dataset_std"] = data_std_csv
    if dataset_id is not None:
        request_body["dataset_id"] = dataset_id
    if dataset_std_id is not None:
        request_body["dataset_std_id"] = dataset_std_id
    return request_body


@typechecked
@check_status_code
def use_model(
    model_id: str,
    method: str,
    data_csv: Optional[str] = None,
    data_std_csv: Optional[str] = None,
    processor: str = "cpu",
    verbose: bool = False,
    **kwargs,
) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/models/{model_id}/{method}"
    headers = _create_headers(verbose=verbose)
    headers["X-Processor"] = processor
    request_body = _use_request_body(data_csv, data_std_csv, **kwargs)
    response = requests.post(url, headers=headers, json=request_body)
    status = response.status_code
    body = _get_response_body(response)
    return status, body


### Asynchronous endpoints ###


@typechecked
@check_status_code
def use_request_model(
    model_id: str,
    method: str,
    data_csv: Optional[str] = None,
    data_std_csv: Optional[str] = None,
    dataset_id: Optional[str] = None,
    dataset_std_id: Optional[str] = None,
    processor: str = "cpu",
    verbose: bool = False,
    **kwargs,
) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/models/{model_id}/async/{method}"
    headers = _create_headers(verbose=verbose)
    headers["X-Processor"] = processor
    request_body = _use_request_body(
        data_csv, data_std_csv, dataset_id, dataset_std_id, **kwargs
    )
    response = requests.post(url, headers=headers, json=request_body)
    status = response.status_code
    body = _get_response_body(response)
    return status, body


@typechecked
@check_status_code
def use_response_model(
    model_id: str,
    method: str,
    process_id: str,
    verbose: bool = False,
) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/models/{model_id}/async/{method}/{process_id}"
    headers = _create_headers(verbose=verbose)
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = _get_response_body(response)
    return status, body


@typechecked
@check_status_code
def get_initial_design(
    priors: str,
    sampling_method: str,
    num_points: int,
    seed: Optional[int] = None,
    verbose: Optional[bool] = False,
) -> dict:
    url = f"{os.getenv('TWINLAB_URL')}/initial_design"
    headers = _create_headers(verbose=verbose)
    response = requests.post(
        url,
        headers=headers,
        json={
            "priors": priors,
            "sample_method": sampling_method,
            "num_points": num_points,
            "seed": seed,
        },
    )
    status = response.status_code
    body = _get_response_body(response)
    return status, body


### ###
