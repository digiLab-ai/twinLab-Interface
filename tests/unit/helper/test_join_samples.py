import pandas as pd
from unittest.mock import patch

import twinlab as tl


def test_join_samples(dataframe_regression):

    # Mock two sample dataframes
    data1 = {
        ("y", "0"): [-2.749357, 0.959321, 0.566999, -0.154889],
        ("y", "1"): [-0.462247, -0.230350, 2.881398, 2.786605],
        ("x", "0"): [0.959321, 0.566999, -0.154889, -0.462247],
        ("x", "1"): [-0.230350, 2.881398, 2.786605, -2.749357],
    }

    df1 = pd.DataFrame(data1)

    data2 = {
        ("y", "0"): [1.0, 2.0, 3.0, 4.0],
        ("y", "1"): [5.0, 6.0, 7.0, 8.0],
        ("x", "0"): [9.0, 10.0, 11.0, 12.0],
        ("x", "1"): [13.0, 14.0, 15.0, 16.0],
    }

    df2 = pd.DataFrame(data2)

    result = tl.join_samples(df1, df2)

    # Check the result
    dataframe_regression.check(result, basename="join_samples_data")
