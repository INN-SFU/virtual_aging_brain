import numpy as np
import matplotlib.pyplot as plt


def plot_ts_stack(
    data,
    x=None,
    scale=0.9,
    lw=0.4,
    color="k",
    title=None,
    labels=None,
    width=48,
    ax=None,
    alpha=1.0,
):
    """
    Plot stacked, normalized time series.

    Parameters
    ----------
    data : array-like, shape (T, N)
        Time series data (T timepoints, N signals).
    x : array-like, optional
        X-axis values. Defaults to np.arange(T).
    scale : float
        Vertical scaling factor for each trace.
    lw : float
        Line width.
    color : str
        Line color.
    title : str, optional
        Plot title.
    labels : sequence, optional
        Y-axis labels for each signal.
    width : float
        Figure width (in inches) if ax is None.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot into.
    alpha : float
        Line transparency.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    data = np.asarray(data)

    data = data - data.mean(axis=0, keepdims=True)

    ranges = data.max(axis=0) - data.min(axis=0)
    max_range = ranges.max()
    if max_range > 0:
        data = data / max_range

    n_time, n_series = data.shape

    if x is None:
        x = np.arange(n_time)

    if ax is None:
        _, ax = plt.subplots(figsize=(width, 0.5 * n_series))

    for i in range(n_series):
        ax.plot(
            x,
            scale * data[:, i] + i,
            color=color,
            lw=lw,
            alpha=alpha,
        )

    ax.set_ylim(-0.5, n_series - 0.5)
    ax.set_yticks(np.arange(n_series))

    if labels is None:
        labels = np.arange(n_series)
    ax.set_yticklabels(labels)

    if title is not None:
        ax.set_title(title)

    ax.autoscale(enable=True, axis="x", tight=True)

    return ax
