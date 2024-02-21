import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict
from enum import Enum
import qis


def plot_vols(vols: pd.DataFrame,
              ax: plt.Subplot = None,
              linestyles: List[str] = None,
              markers: List[str] = None,
              label_x_y: Dict[str, Tuple[float, float]] = None,
              **kwargs
              ) -> None:

    kwargs = qis.update_kwargs(kwargs,
                               dict(ncol=3, legend_loc='upper center',
                                    xvar_format='{:,.2f}',
                                    yvar_format='{:.0%}',
                                    markersize=10))
    if ax is None:
        with sns.axes_style("darkgrid"):
            fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)

    markers = markers or ["o"]*len(vols.columns)
    linestyles = linestyles or ['-']*len(vols.columns)
    qis.plot_line(df=vols,
                  linestyles=linestyles,
                  markers=markers,
                  ylabel='Implied Vol',
                  ax=ax,
                  **kwargs)

    if label_x_y is not None:
        qis.add_scatter_points(ax=ax, label_x_y=label_x_y, linewidth=10)


def map_deltas_to_str(bsm_deltas: np.ndarray, delta_str_format: str = '0.2f') -> List[str]:
    """
    map deltas to str of 0.2f
    deltas below 0.05 are mapped as 0.04
    """
    slice_index = []
    index_str = np.empty_like(bsm_deltas, dtype=str)
    for idx, x in enumerate(bsm_deltas):
        if np.abs(x) < 0.05:
            x_str = f"{x:0.4f}"
        else:
            x_str = f"{x:{delta_str_format}}"
        # check for non overlaps
        if idx > 0:
            if x_str == index_str[idx - 1]:
                if x < 0.0:  # decrease previous delta
                    slice_index[idx-1] = f"{bsm_deltas[idx-1]:0.4f}"
                else:
                    x_str = f"{x:0.4f}"
        slice_index.append(x_str)
    return slice_index
