import pandas as pd

import twinlab as tl


def test_get_sample(dataframe_regression):

    # Mock sample data
    data = {
        ("y", "0"): [-2.749357, 0.959321, 0.566999, -0.154889],
        ("y", "1"): [-0.462247, -0.230350, 2.881398, 2.786605],
        ("x", "0"): [0.959321, 0.566999, -0.154889, -0.462247],
        ("x", "1"): [-0.230350, 2.881398, 2.786605, -2.749357],
    }

    df = pd.DataFrame(data)

    result = tl.get_sample(df, key=0)

    # Check the result
    dataframe_regression.check(result, basename="get_sample_data")
