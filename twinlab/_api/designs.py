import os
from typing import Optional, Tuple

import requests
from typeguard import typechecked

from ._utils import check_status_code, create_headers


@typechecked
@check_status_code
def post_design(
    priors: str,
    sampling_method: str,
    num_points: int,
    seed: Optional[int] = None,
) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/designs"
    headers = create_headers()
    body = {
        "priors": priors,
        "sample_method": sampling_method,
        "num_points": num_points,
        "seed": seed,
    }
    response = requests.post(
        url,
        headers=headers,
        json=body,
    )
    status = response.status_code
    body = response.json()
    return status, body


@typechecked
@check_status_code
def get_design(process_id: str) -> Tuple[int, dict]:
    url = f"{os.getenv('TWINLAB_URL')}/designs/{process_id}"
    headers = create_headers()
    response = requests.get(url, headers=headers)
    status = response.status_code
    body = response.json()
    return status, body
