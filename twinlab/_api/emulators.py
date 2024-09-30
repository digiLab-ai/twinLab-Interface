import os
from typing import Tuple

import requests
from typeguard import typechecked

from ._utils import check_status_code, create_headers

### General emulator stuff ###


@typechecked
@check_status_code
def get_emulators() -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/emulators"
    headers = create_headers()
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = response.json()
    return status, body


@typechecked
@check_status_code
def get_emulators_statuses() -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/emulators-statuses"
    headers = create_headers()
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = response.json()
    return status, body


@typechecked
@check_status_code
def get_emulator_parameters(emulator_id: str) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/emulators/{emulator_id}/parameters"
    headers = create_headers()
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = response.json()
    return status, body


@typechecked
@check_status_code
def get_emulator_data(emulator_id: str, dataset_type: str) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/emulators/{emulator_id}/data/{dataset_type}"
    headers = create_headers()
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = response.json()
    return status, body


@typechecked
@check_status_code
def get_emulator_summary(emulator_id: str) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/emulators/{emulator_id}/summary"
    headers = create_headers()
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = response.json()
    return status, body


@typechecked
@check_status_code
def delete_emulator(emulator_id: str) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/emulators/{emulator_id}"
    headers = create_headers()
    response = requests.delete(url, headers=headers)
    status = response.status_code
    body = response.json()
    return status, body


### ###

### Training endpoints ###


@typechecked
@check_status_code
def post_emulator(
    emulator_id: str, emulator_params: dict, training_params: dict, processor: str
) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/emulators/{emulator_id}"
    headers = create_headers()
    headers["X-Processor"] = processor
    request_body = {
        "emulator_params": emulator_params,
        "training_params": training_params,
    }
    response = requests.post(url, headers=headers, json=request_body)
    status = response.status_code
    body = response.json()
    return status, body


@typechecked
@check_status_code
def get_emulator_status(emulator_id: str) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/emulators/{emulator_id}/status"
    headers = create_headers()
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = response.json()
    return status, body


### ###

### Emulator method endpoints ###


@typechecked
@check_status_code
def post_emulator_update(emulator_id: str, params: dict) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/emulators/{emulator_id}/update"
    headers = create_headers()
    response = requests.post(url, headers=headers, json=params)
    status = response.status_code
    body = response.json()
    return status, body


@typechecked
@check_status_code
def post_emulator_score(emulator_id: str, params: dict) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/emulators/{emulator_id}/score"
    headers = create_headers()
    response = requests.post(url, headers=headers, json=params)
    status = response.status_code
    body = response.json()
    return status, body


@typechecked
@check_status_code
def post_emulator_benchmark(emulator_id: str, params: dict) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/emulators/{emulator_id}/benchmark"
    headers = create_headers()
    response = requests.post(url, headers=headers, json=params)
    status = response.status_code
    body = response.json()
    return status, body


@typechecked
@check_status_code
def post_emulator_predict(emulator_id: str, params: dict) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/emulators/{emulator_id}/predict"
    headers = create_headers()
    response = requests.post(url, headers=headers, json=params)
    status = response.status_code
    body = response.json()
    return status, body


@typechecked
@check_status_code
def post_emulator_sample(emulator_id: str, params: dict) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/emulators/{emulator_id}/sample"
    headers = create_headers()
    response = requests.post(url, headers=headers, json=params)
    status = response.status_code
    body = response.json()
    return status, body


@typechecked
@check_status_code
def post_emulator_recommend(emulator_id: str, params: dict) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/emulators/{emulator_id}/recommend"
    headers = create_headers()
    response = requests.post(url, headers=headers, json=params)
    status = response.status_code
    body = response.json()
    return status, body


@typechecked
@check_status_code
def post_emulator_calibrate(emulator_id: str, params: dict) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/emulators/{emulator_id}/calibrate"
    headers = create_headers()
    response = requests.post(url, headers=headers, json=params)
    status = response.status_code
    body = response.json()
    return status, body


@typechecked
@check_status_code
def post_emulator_maximize(emulator_id: str, params: dict) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/emulators/{emulator_id}/maximize"
    headers = create_headers()
    response = requests.post(url, headers=headers, json=params)
    status = response.status_code
    body = response.json()
    return status, body


### ###

### Process endpoints ###


@typechecked
@check_status_code
def get_emulator_processes(emulator_id: str) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/emulators/{emulator_id}/processes"
    headers = create_headers()
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = response.json()
    return status, body


@typechecked
@check_status_code
def get_emulator_processes_statuses(emulator_id: str) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/emulators/{emulator_id}/processes-statuses"
    headers = create_headers()
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = response.json()
    return status, body


@typechecked
@check_status_code
def get_emulator_process(
    emulator_id: str,
    process_id: str,
) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/emulators/{emulator_id}/processes/{process_id}"
    headers = create_headers()
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = response.json()
    return status, body


@typechecked
@check_status_code
def delete_emulator_process(emulator_id: str, process_id: str) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/emulators/{emulator_id}/processes/{process_id}"
    headers = create_headers()
    response = requests.delete(url, headers=headers)
    status = response.status_code
    body = response.json()
    return status, body


### ###

### Exporting Emulators ###


@typechecked
@check_status_code
def post_emulator_torchscript(emulator_id: str, params: dict) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/emulators/{emulator_id}/torchscript"
    headers = create_headers()
    response = requests.post(url, headers=headers, json=params)
    status = response.status_code
    body = response.json()
    return status, body


@typechecked
@check_status_code
def get_emulator_torchscript(emulator_id: str, process_id: str) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/emulators/{emulator_id}/torchscript/{process_id}"
    headers = create_headers()
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = response.json()
    return status, body


### ###
