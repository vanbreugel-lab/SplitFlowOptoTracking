"""
plot_bar.py
-----------
Helper function for plotting a single, outline-free bar.
"""


def plot_single_bar(ax, height, color, width=0.6, bottom=0, **kwargs):
    """
    Plot a single bar with no edge/outline.

    Parameters
    ----------
    ax      : matplotlib Axes
    height  : height of the bar
    color   : fill color (any matplotlib color spec)
    width   : bar width (default 0.6)
    bottom  : bar base y-value (default 0, useful for stacking)
    **kwargs: forwarded to ax.bar()

    Returns
    -------
    bar : BarContainer
    """
    bar = ax.bar(
        0, height,
        width=width,
        bottom=bottom,
        color=color,
        edgecolor="none",
        linewidth=0,
        **kwargs,
    )
    return bar
