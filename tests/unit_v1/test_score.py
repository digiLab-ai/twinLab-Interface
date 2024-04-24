import io
from unittest.mock import patch

import pandas as pd

import twinlab as tl

EXPECTED_RESULT = pd.DataFrame(
    [[186340.2500700946, 8353444.886127352]],
    columns=["Number of packs sold", "Profit [GBP]"],
)


@patch("twinlab.client._use_campaign")
def test_score_campaign_use(mock_use_campaign):
    """
    Test for twinlab.client.score_campaign()

    Requirements
    ------------
        method calls _use_campaign
    """

    mock_use_campaign.return_value = io.StringIO(EXPECTED_RESULT.to_csv(index=False))

    _ = tl.client.score_campaign(
        campaign_id="score",
        combined_score=False,
        processor="cpu",
        verbose=False,
        debug=False,
    )

    mock_use_campaign.assert_called_once()


@patch("twinlab.client._use_campaign")
def test_score_campaign_out(mock_use_campaign):
    """
    Test for twinlab.client.score_campaign()

    Requirements
    ------------
        method returns the correct (self-defined) list
    """

    # Define the expected output
    # expected_list = [186340.2500700946, 8353444.886127352]
    mock_use_campaign.return_value = io.StringIO(EXPECTED_RESULT.to_csv(index=False))
    # Define the expected DataFrame
    # expected_df = pd.DataFrame(expected_list, columns=["Score"])

    result_df = tl.client.score_campaign(
        campaign_id="score",
        combined_score=False,
        processor="cpu",
        verbose=False,
        debug=False,
    )

    pd.testing.assert_frame_equal(result_df, EXPECTED_RESULT)
