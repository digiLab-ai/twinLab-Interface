import os
import requests
from typing import Optional, Dict

from typeguard import typechecked

from ._utils import check_status_code, create_headers


@typechecked
@check_status_code
def get_projects() -> Dict[str, str]:
    headers = create_headers()
    url = f"{os.getenv('TWINLAB_URL')}/projects"
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = response.json()
    return status, body


@typechecked
@check_status_code
def post_project(
    project_name: str, organization_name: Optional[str] = None
) -> Dict[str, str]:
    headers = create_headers()
    url = f"{os.getenv('TWINLAB_URL')}/projects"
    data = {"project_name": project_name}
    response = requests.post(url, headers=headers, json=data)
    status = response.status_code
    body = response.json()
    return status, body


@typechecked
@check_status_code
def delete_project(project_id: str) -> Dict[str, str]:
    headers = create_headers()
    url = f"{os.getenv('TWINLAB_URL')}/projects/{project_id}"
    response = requests.delete(url, headers=headers)
    status = response.status_code
    body = response.json()
    return status, body


@typechecked
@check_status_code
def post_project_members_account(
    project_id: str, account_id: str, role: str
) -> Dict[str, str]:
    headers = create_headers()
    url = (
        f"{os.getenv('TWINLAB_URL')}/projects/{project_id}/members/{account_id}/{role}"
    )
    response = requests.post(url, headers=headers)
    status = response.status_code
    body = response.json()
    return status, body


@typechecked
@check_status_code
def delete_project_members_account(project_id: str, account_id: str) -> Dict[str, str]:
    headers = create_headers()
    url = f"{os.getenv('TWINLAB_URL')}/projects/{project_id}/members/{account_id}"
    response = requests.delete(url, headers=headers)
    status = response.status_code
    body = response.json()
    return status, body
