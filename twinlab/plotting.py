import math

from typeguard import typechecked

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

DIGILAB_COLORS = {
    "dark_blue": "#162448",
    "light_blue": "#009FE3",
    "green": "#7DB928",
    "cement": "#EEEEEE",
}
DIGILAB_CMAP = LinearSegmentedColormap.from_list(
    "digiLab", list(DIGILAB_COLORS.values())
)


@typechecked
def plot(
    x_axis: str,
    y_axis: str,
    df_X: pd.DataFrame,
    df_mean: pd.DataFrame,
    df_std: pd.DataFrame,
    label: str = "Emulator prediction",
    color: str = DIGILAB_COLORS["dark_blue"],
) -> plt:

    X = df_X[x_axis]
    y, dy = df_mean[y_axis], df_std[y_axis]
    nsig = [1, 2]
    alpha = 0.5
    plt.subplots()
    plt.plot(X, y, color=color, label=label)
    for sig in nsig:  # Loop over the number of sigmas to plot
        plt.fill_between(
            X, y - sig * dy, y + sig * dy, facecolor=color, linewidth=0.0, alpha=alpha
        )
    plt.xlabel(x_axis)
    plt.xlim(X.min(), X.max())
    plt.ylabel(y_axis)
    plt.legend()
    return plt


@typechecked
def heatmap(
    x_axis: str,
    y_axis: str,
    z_axis: str,
    df_X: pd.DataFrame,
    df_mean: pd.DataFrame,
    cmap=DIGILAB_CMAP,
) -> plt:

    n = int(math.sqrt(len(df_X)))
    xmin, xmax = df_X[x_axis].min(), df_X[x_axis].max()
    ymin, ymax = df_X[y_axis].min(), df_X[y_axis].max()
    extent = [xmin, xmax, ymin, ymax]
    plt.subplots()
    plt.imshow(
        df_mean[z_axis].values.reshape(n, n),
        cmap=cmap,
        origin="lower",  # Lower left corner of (0,0) is correct for data
        extent=extent,  # Sets axes labels according to data
        aspect=xmax / ymax,  # Correct aspect ratio
        interpolation="bicubic",  # Nice default smoothing
    )
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.colorbar(label=z_axis)
    return plt
