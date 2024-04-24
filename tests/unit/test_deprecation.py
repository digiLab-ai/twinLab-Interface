# Third-party imports
import pytest
from unittest.mock import patch

# Local imports
import twinlab as tl


@patch("twinlab.api.summarise_dataset")
def test_deprecation(mock_summarise_dataset):

    mock_summarise_dataset.return_value = 200, {"dataset_summary": "test_summary"}

    # Use pytest.warns to catch the FutureWarning
    with pytest.warns(DeprecationWarning):
        tl.query_dataset("test_dataset")
