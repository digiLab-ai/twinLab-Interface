import os
from typing import Optional

from .._version import __version__


def create_headers(processor: Optional[str] = "cpu") -> dict[str, str]:
    twinlab_user = os.getenv("TWINLAB_USER")
    if not twinlab_user:  # Covers None and empty string
        raise ValueError(
            "TWINLAB_USER environment variable not set. Please set this in a .env file or use tl.set_user()."
        )
    twinlab_api_key = os.getenv("TWINLAB_API_KEY")
    if not twinlab_api_key:  # Covers None and empty string
        raise ValueError(
            "TWINLAB_API_KEY environment variable not set. Please set this in a .env file or use tl.set_api_key()."
        )
    headers = {
        "X-User": twinlab_user,
        "X-API-Key": twinlab_api_key,
        "X-Processor": processor,
        "X-Language": "python",
        "X-Client-Version": __version__,
    }
    return headers


def check_status_code(func):
    """
    This is wrapper function that is applied to all endpoints.
    If a non 2xx status code is returned it captures the error message and presents
    it in a slightly nicer format.
    """

    def wrapper(*args, **kwargs):
        status, body = func(*args, **kwargs)
        if str(status).startswith("2"):  # Success
            return status, body
        else:
            if status == 422:  # Special treatment for 422 generated by pydantic
                try:
                    message = body["detail"][0]["msg"]
                except (KeyError, IndexError):
                    message = body["detail"]
            else:
                if "detail" in body:
                    message = body["detail"]
                else:
                    message = body
            raise Exception(f"Error Code: {status} - {message}")

    return wrapper
