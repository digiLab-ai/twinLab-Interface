import io
from unittest.mock import patch

import pandas as pd
import pytest

import twinlab as tl


@patch("twinlab.client._use_campaign")
def test_optimise_use(mock_use_campaign):
    """
    Test for twinlab.client.optimise_campaign()

    Requirements
    ------------
        method calls _use_campaign
    """

    with pytest.raises(TypeError):
        _ = tl.client.optimise_campaign(
            campaign_id="optimise",
            num_points=1,
            processor="cpu",
            verbose=False,
            debug=False,
        )

    mock_use_campaign.assert_called_once()


@patch("twinlab.client._use_campaign")
def test_optimise_out(mock_use_campaign):
    """
    Test for twinlab.client.optimise_campaign()

    Requirements
    ------------
        method returns the correct (self-defined) DataFrame
    """

    io_string = "Pack price [GBP],Number of biscuits per pack\n2.1577753355771727,16.0\n2.1411597690963453,17.028095712314485\n2.3,16.0\n2.3,17.17760751248427\n2.3,18.2660188939969\n"
    mock_use_campaign.return_value = io.StringIO(io_string)

    # Define the expected DataFrame
    expected_df = pd.read_csv(io.StringIO(io_string))

    result = tl.client.optimise_campaign(
        campaign_id="optimise",
        num_points=5,
        processor="cpu",
        verbose=False,
        debug=False,
    )

    pd.testing.assert_frame_equal(result, expected_df)
