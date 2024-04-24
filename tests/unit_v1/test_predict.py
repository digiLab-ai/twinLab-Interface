import io
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import twinlab as tl


@patch("twinlab.client._use_campaign")
def test_predict_use(mock_use_campaign):
    """
    Test for twinlab.client.predict_campaign()

    Requirements
    ------------
        method calls:
            api.predict_request_model
            api.predict_response_model
    """

    with pytest.raises(TypeError):
        _ = tl.client.predict_campaign(
            filepath_or_df="data_path.csv",
            campaign_id="predict",
            sync=True,
            processor="cpu",
            verbose=False,
            debug=False,
        )

    mock_use_campaign.assert_called_once()


# @patch("twinlab.client._get_csv_string")
# @patch("twinlab.api.predict_request_model")
# @patch("twinlab.client._get_value_from_body")
# @patch("twinlab.api.predict_response_model")
# def test_predict_out(
#     mock_get_csv_string,
#     mock_predict_request_model,
#     mock_get_value_from_body,
#     mock_predict_response_model,
# ):
#     """
#     Test for twinlab.client.optimise_campaign()

#     Requirements
#     ------------
#         method returns the correct (self-defined) DataFrame
#     """

#     mock_predict_request_model.return_value = {"process_id": "test"}
#     mock_get_value_from_body.return_value = "test"
#     io_string = "y,y_stdev\n2.1577753355771727,16.0\n2.1411597690963453,17.028095712314485\n2.3,16.0\n2.3,17.17760751248427\n2.3,18.2660188939969\n"
#     mock_predict_response_model.return_value = {
#         "process_id": "test",
#         "dataframe": io_string,
#     }

#     mean, stdev = tl.client.predict_campaign(
#         filepath_or_df="data_path.csv",
#         campaign_id="predict",
#         processor="cpu",
#         verbose=False,
#         debug=False,
#     )
#     # Define the expected DataFrame
#     expected_df = pd.read_csv(io.StringIO(io_string))
#     np.testing.assert_array_equal(mean.values.flatten(), expected_df["y"].values)
#     np.testing.assert_array_equal(stdev.values.flatten(), expected_df["y_stdev"].values)
