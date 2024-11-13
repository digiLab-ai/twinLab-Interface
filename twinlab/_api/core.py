import os
from typing import Dict

import requests
from typeguard import typechecked

from ._utils import check_status_code, create_headers


@typechecked
@check_status_code
def get_user() -> Dict[str, str]:
    headers = create_headers()
    url = f"{os.getenv('TWINLAB_URL')}/user"
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = response.json()
    return status, body


@typechecked
@check_status_code
def get_account(account_email: str) -> Dict[str, str]:
    headers = create_headers()
    url = f"{os.getenv('TWINLAB_URL')}/accounts/{account_email}"
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = response.json()
    return status, body


@typechecked
@check_status_code
def get_versions() -> Dict[str, str]:
    headers = create_headers()
    url = f"{os.getenv('TWINLAB_URL')}/versions"
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = response.json()
    return status, body
