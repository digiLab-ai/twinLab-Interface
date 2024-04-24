from unittest.mock import patch

import pandas as pd
import pytest
import twinlab as tl


@pytest.fixture
def mock_generate_upload_url_response():
    return 200, {"url": "url"}


@pytest.fixture
def mock_process_upload_df_to_url_response():
    return 200, {"message": "Dataset dataset was uploaded."}


@pytest.fixture
def mock_process_uploaded_success_response():
    return 200, {"message": "Dataset dataset was uploaded."}


@pytest.fixture
def mock_df():
    return pd.DataFrame(
        {
            "Sunlight [hours/day]": [6, 8, 10, 9, 7],
            "Water [times/week]": [2.0, 3.0, 3.0, 4.0, 2.0],
        }
    )


@patch("twinlab.api.process_uploaded_dataset")
@patch("twinlab.utils.upload_dataframe_to_presigned_url")
@patch("twinlab.api.generate_upload_url")
def test_success(
    mock_generate_upload_url,
    mock_upload_dataframe_to_presigned_url,
    mock_process_uploaded_dataset,
    mock_generate_upload_url_response,
    mock_process_upload_df_to_url_response,
    mock_process_uploaded_success_response,
    mock_df,
    capsys,
):
    # Arrange
    dataset = tl.Dataset("dataset")
    mock_generate_upload_url.return_value = mock_generate_upload_url_response
    mock_upload_dataframe_to_presigned_url.return_value = (
        mock_process_upload_df_to_url_response
    )
    mock_process_uploaded_dataset.return_value = mock_process_uploaded_success_response

    # Act
    dataset.upload(mock_df, verbose=True)

    # Capture the printed output
    captured = capsys.readouterr()

    # Assert
    assert "Dataset dataset was uploaded." in captured.out
