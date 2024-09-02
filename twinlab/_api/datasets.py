import os
from typing import List, Tuple

import requests
from typeguard import typechecked

from ._utils import check_status_code, create_headers

### Uploading via URL endpoints ###


@typechecked
@check_status_code
def get_dataset_upload_url(dataset_id: str) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/datasets/{dataset_id}/upload-url"
    headers = create_headers()
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = response.json()
    return status, body


@typechecked
@check_status_code
def get_dataset_temporary_upload_url(dataset_id: str) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/datasets/{dataset_id}/temporary-upload-url"
    headers = create_headers()
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = response.json()
    return status, body


### ###

### Dataset endpoints ###


@typechecked
@check_status_code
def get_datasets() -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/datasets"
    headers = create_headers()
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = response.json()
    return status, body


@typechecked
@check_status_code
def post_dataset(dataset_id: str, data_csv: str) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/datasets/{dataset_id}"
    headers = create_headers()
    request_body = {"dataset": data_csv}
    response = requests.post(url, headers=headers, json=request_body)
    status = response.status_code
    body = response.json()
    return status, body


@typechecked
@check_status_code
def get_dataset(dataset_id: str) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/datasets/{dataset_id}"
    headers = create_headers()
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = response.json()
    return status, body


@typechecked
@check_status_code
def post_dataset_append(dataset_id: str, new_dataset_id: str) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/datasets/{dataset_id}/append"
    headers = create_headers()
    request_body = {"dataset_id": new_dataset_id}
    response = requests.post(url, headers=headers, json=request_body)
    status = response.status_code
    body = response.json()
    return status, body


@typechecked
@check_status_code
def post_dataset_summary(dataset_id: str) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/datasets/{dataset_id}/summary"
    headers = create_headers()
    response = requests.post(url, headers=headers)
    status = response.status_code
    body = response.json()
    return status, body


@typechecked
@check_status_code
def get_dataset_summary(dataset_id: str) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/datasets/{dataset_id}/summary"
    headers = create_headers()
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = response.json()
    return status, body


@typechecked
@check_status_code
def post_dataset_analysis(dataset_id: str, columns: List[str]) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/datasets/{dataset_id}/analysis"
    headers = create_headers()
    query_params = {"columns": columns}
    response = requests.post(url, headers=headers, json=query_params)
    status = response.status_code
    body = response.json()
    return status, body


@typechecked
@check_status_code
def get_dataset_process(dataset_id: str, process_id: str) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/datasets/{dataset_id}/processes/{process_id}"
    headers = create_headers()
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = response.json()
    return status, body


@typechecked
@check_status_code
def delete_dataset(dataset_id: str) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/datasets/{dataset_id}"
    headers = create_headers()
    response = requests.delete(url, headers=headers)
    status = response.status_code
    body = response.json()
    return status, body


### ###

### Example dataset endpoints ###


@typechecked
@check_status_code
def get_example_datasets() -> dict:
    url = f"{os.getenv('TWINLAB_URL')}/example-datasets"
    headers = create_headers()
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = response.json()
    return status, body


@typechecked
@check_status_code
def get_example_dataset(dataset_id: str) -> dict:
    url = f"{os.getenv('TWINLAB_URL')}/example-datasets/{dataset_id}"
    headers = create_headers()
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = response.json()
    return status, body


### ###
