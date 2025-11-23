"""
Minimal seaborn stub for environments where the real dependency is unavailable.
Only implements the `heatmap` function needed by final.py.
"""

from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np


def heatmap(
    data,
    annot: bool = False,
    fmt: str = "d",
    cmap: str = "Blues",
    xticklabels: Optional[Iterable[str]] = None,
    yticklabels: Optional[Iterable[str]] = None,
    cbar_kws: Optional[dict] = None,
    annot_kws: Optional[dict] = None,
    **kwargs,
):
    """Render a simple heatmap using matplotlib."""
    data = np.array(data)
    fig = plt.gcf()
    ax = plt.gca()
    im = ax.imshow(data, cmap=cmap, aspect="auto")

    if cbar_kws is None:
        cbar_kws = {}
    fig.colorbar(im, ax=ax, **cbar_kws)

    if xticklabels is not None:
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticks(range(len(yticklabels)))
        ax.set_yticklabels(yticklabels)

    if annot:
        if annot_kws is None:
            annot_kws = {}
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(
                    j,
                    i,
                    format(data[i, j], fmt),
                    ha="center",
                    va="center",
                    **annot_kws,
                )

    return ax
