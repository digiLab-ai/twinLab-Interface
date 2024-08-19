import math
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from typeguard import typechecked

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
    label: Optional[str] = "Emulator",
    color: str = DIGILAB_COLORS["light_blue"],
    figsize: Tuple[float, float] = (6.4, 4.8),
) -> plt:

    X = df_X[x_axis]
    y, dy = df_mean[y_axis], df_std[y_axis]
    plt.subplots(figsize=figsize)
    nsig = [1, 2]
    alpha = 0.5
    plt.plot(X, y, color=color, label=label)
    for sig in nsig:  # Loop over the number of sigmas to plot
        plt.fill_between(
            X,
            y - sig * dy,
            y + sig * dy,
            facecolor=color,
            alpha=alpha,
            linewidth=0.0,
        )
    plt.xlabel(x_axis)
    plt.xlim(X.min(), X.max())
    plt.ylabel(y_axis)
    if label is not None:
        plt.legend()
    return plt


@typechecked
def blur(
    x_axis: str,
    y_axis: str,
    df_X: pd.DataFrame,
    df_mean: pd.DataFrame,
    df_std: pd.DataFrame,
    label: Optional[str] = "Emulator",
    color: str = DIGILAB_COLORS["light_blue"],
    figsize: Tuple[float, float] = (6.4, 4.8),
) -> plt:

    X = df_X[x_axis]
    y, dy = df_mean[y_axis], df_std[y_axis]
    plt.subplots(figsize=figsize)
    sig_max = 3.0
    nsig = 100
    sigs = np.linspace(0.0, sig_max, nsig)
    alpha = 0.8
    previous_alpha = 0.0
    for i in range(nsig - 1, 0, -1):
        sig = sigs[i]
        target_alpha = alpha * np.exp(-0.5 * sig**2)
        _alpha = 1.0 - (1.0 - target_alpha) / (1.0 - previous_alpha)
        plt.fill_between(
            X,
            y - sig * dy,
            y + sig * dy,
            facecolor=color,
            alpha=_alpha,
            edgecolor="none",
            linewidth=0.0,
        )
        previous_alpha = target_alpha
    plt.fill_between(  # For label only
        X,
        np.nan,
        np.nan,
        facecolor=color,
        alpha=alpha,
        edgecolor="none",
        linewidth=0.0,
        label=label,
    )
    plt.xlabel(x_axis)
    plt.xlim(X.min(), X.max())
    plt.ylabel(y_axis)
    if label is not None:
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
    figsize: Tuple[float, float] = (6.4, 4.8),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> plt:

    n = int(math.sqrt(len(df_X)))
    xmin, xmax = df_X[x_axis].min(), df_X[x_axis].max()
    ymin, ymax = df_X[y_axis].min(), df_X[y_axis].max()
    extent = [xmin, xmax, ymin, ymax]
    plt.subplots(figsize=figsize)
    plt.imshow(
        df_mean[z_axis].values.reshape(n, n),
        cmap=cmap,
        origin="lower",  # Lower left corner of (0,0) is correct for data
        extent=extent,  # Sets axes labels according to data
        aspect=xmax / ymax,  # Correct aspect ratio
        interpolation="bicubic",  # Nice default smoothing
        vmin=vmin,
        vmax=vmax,
    )
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.colorbar(label=z_axis)
    return plt
