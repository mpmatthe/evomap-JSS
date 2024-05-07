"""
Functions to draw maps.
"""

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from cycler import cycler

DEFAULT_BUBBLE_SIZE = 25
DEFAULT_FONT_SIZE = 10

title_fontdict_large = {'size': 20, 'family': 'Arial'}
title_fontdict = {'size': 18, 'family': 'Arial'}
text_fontdict = {'size': 14, 'family': 'Arial'}
axis_label_fontdict = {'size': 16, 'family': 'Arial'}

def format_tick_labels(x, pos):
    return '{0:.2f}'.format(x)

def init_params(custom_params=None):
    """
    Initialize plot aesthetics.
    """
    base_style = {
        "axes.prop_cycle": cycler('color', ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                                            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                                            "#bcbd22", "#17becf"]),
        "axes.linewidth": 1,
        "axes.titlesize": 22,
        "axes.labelsize": 16,
        "font.family": 'Arial',
        "axes.edgecolor": "black",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.major.size": 0.2,
        "xtick.minor.size": 0.1,
        "ytick.major.size": 0.2,
        "ytick.minor.size": 0.1,
        "axes.grid": False,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        "grid.color": "black",
        "savefig.facecolor": "w",
        "savefig.transparent": False,
        "savefig.bbox": "tight",
        "savefig.format": "png"
    }

    if custom_params:
        base_style.update(custom_params)

    mpl.rcParams.update(base_style)


def style_axes(ax, show_axes=True, show_box=True, show_grid=False, axes_at_origin=False):
    """
    Style axes of a matplotlib Axes object according to the specified options.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to apply styling to.
    show_axes : bool, optional
        If True, display the axes lines and labels; otherwise, hide them. Default is True.
    show_box : bool, optional
        If True, display the bounding box (spines) around the plot. If False, hide the top and right spines.
        The visibility of left and bottom spines is controlled by `show_axes`. Default is True.
    show_grid : bool, optional
        If True, display grid lines on the plot. Grid lines are only displayed if `show_axes` is also True.
        Default is False.
    axes_at_origin : bool, optional
        If True, move the 'left' and 'bottom' axes to intersect at the origin (0,0) point. This setting overrides
        the visibility settings for 'top' and 'right' spines, setting them to False regardless of `show_box`.
        Default is False.

    """
    # Handling the visibility of the axis lines and labels
    ax.xaxis.set_visible(show_axes)
    ax.yaxis.set_visible(show_axes)

    # Handling grid lines
    ax.grid(show_grid and show_axes)  # Grid only if show_axes is True and show_grid is requested

    # Adjusting axis spines (the box)
    if show_box:
        # Ensuring all spines are visible
        for spine in ax.spines.values():
            spine.set_visible(True)
    else:
        # Setting spines visibility based on axes visibility when the box isn't requested
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(show_axes)
        ax.spines['bottom'].set_visible(show_axes)

    # Positioning axes at the origin
    if axes_at_origin and show_axes:
        # Move axes to zero point
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    elif not axes_at_origin:
        # Reset spines to default positions
        ax.spines['left'].set_position(('outward', 0))
        ax.spines['bottom'].set_position(('outward', 0))

    # Setting labels
    ax.set_xlabel("Dimension 1" if show_axes else "")
    ax.set_ylabel("Dimension 2" if show_axes else "")

    # Ensure ticks are shown only when axes are visible
    ax.tick_params(axis='x', which='both', bottom=show_axes, labelbottom=show_axes)
    ax.tick_params(axis='y', which='both', left=show_axes, labelleft=show_axes)

    # Setting default limits if no axes or box are shown to prevent zooming effect
    if not show_axes and not show_box:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)



def draw_map(X, label=None, color=None, size=None, inclusions=None, zoom_on_cluster=None, highlighted_labels=None, 
             show_box=True, show_grid=False, show_axes=False, axes_at_origin=False, show_legend=False,
             cmap=None, filename=None, ax=None, fig_size=None, 
             title=None, rotate_labels=0, scatter_kws={}, fontdict=None, rcparams=None):
    """
    Plot a scatter map with optional labels, coloring, and sizing.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data points to plot. n_features should be 1 or 2.
    label : array-like, optional
        Labels for each data point.
    color : array-like, optional
        Colors or group identifiers for each data point. If None, all points will have the same color.
    size : array-like, optional
        Sizes for each data point. If None, a default size is used.
    inclusions : array-like of bool, optional
        Boolean array to select which points are included in the plot.
    zoom_on_cluster : int or string, optional
        Cluster identifier to zoom in on specific cluster data points.
    highlighted_labels : list, optional
        Labels to be highlighted on the plot.
    show_box : bool, optional
        If True, show a box around the plot. Default is True.
    show_grid : bool, optional
        If True, show grid lines on the plot. Default is False.
    show_axes : bool, optional
        If True, show the axes of the plot. Default is False.
    axes_at_origin : bool, optional
        If True, draw axes lines through the origin. Default is False.
    show_legend : bool, optional
        If True, display a legend on the plot. Default is False.
    cmap : str or Colormap, optional
        Colormap to use for coloring the points. If None, a default colormap is used.
    filename : str, optional
        Path to save the figure file. If None, the figure is not saved.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes for the plot. If None, a new figure and axes are created.
    fig_size : tuple, optional
        Size of the figure to create. Ignored if `ax` is not None.
    title : str, optional
        Title of the plot.
    rotate_labels : int, optional
        Angle to rotate the labels. Default is 0.
    scatter_kws : dict, optional
        Additional keyword arguments to pass to the scatter plot function.
    fontdict : dict, optional
        Font dictionary for the labels. If None, a default fontdict is used.
    rcparams : dict, optional
        Dictionary to update matplotlib's rcParams for customizing plots.

    Returns
    -------
    matplotlib.figure.Figure
        Only if `ax` is None, the figure containing the plot is returned.
    """

    n_samples = len(X)
    X = np.atleast_2d(X)  # Ensure X is at least 2D
    if X.shape[1] == 1:
        X = np.hstack([X, np.zeros_like(X)])  # Handle 1D data by adding a dummy second dimension

    if color is None:
        color = np.zeros(n_samples)
    color = np.asarray(color)

    # Prepare dataframe for plotting
    df_data = pd.DataFrame(X, columns=['x', 'y'])
    df_data['color'] = pd.Categorical(color)
    if label is not None:
        df_data['label'] = label
    if size is not None:
        df_data['size'] = size
    else:
        df_data['size'] = DEFAULT_BUBBLE_SIZE

    if inclusions is not None:
        df_data = df_data[inclusions == 1]

    if zoom_on_cluster is not None:
        df_data = df_data[df_data['color'] == zoom_on_cluster]

    # Handle colormap
    unique_colors = df_data['color'].cat.categories
    if cmap is None:
        cmap = "tab10" if len(unique_colors) <= 10 else "tab20"
    cmap = plt.get_cmap(cmap)
    df_data['color'] = [cmap(i) for i in df_data['color'].cat.codes]

    init_params(rcparams)

    # Setup plot
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size or (5, 5))
        return_fig = True
    else:
        return_fig = False

    # Plotting
    scatter_args = {'edgecolors': 'none', 'alpha': 0.6}
    scatter_args.update(scatter_kws)
    for key, grp in df_data.groupby('color'):
        ax.scatter(grp['x'], grp['y'], c=[key] * len(grp), s=grp['size'], **scatter_args)
    
    # Handling labels
    fontdict = fontdict or text_fontdict.copy()
    for _, row in df_data.iterrows():
        if highlighted_labels and row['label'] in highlighted_labels:
            ax.text(row['x'], row['y'], row['label'], fontdict={'weight': 'bold', **fontdict}, rotation=rotate_labels)
        elif label is not None:
            ax.text(row['x'], row['y'], row['label'], fontdict=fontdict, rotation=rotate_labels)

    style_axes(ax, show_axes, show_box, show_grid, axes_at_origin)

    if title:
        ax.set_title(title, fontdict=title_fontdict)

    if show_legend:
        ax.legend(title='Cluster')

    # Save or show plot
    if filename:
        plt.savefig(filename, dpi=300, format='png', bbox_inches='tight')
    plt.close()

    if return_fig:
        return fig

def normalize_dhat(d_hat, n_samples):
    """ Normalize dissimilarity predictions. """
    return d_hat * np.sqrt((n_samples * (n_samples - 1) / 2) / np.sum(d_hat**2))

def draw_shepard_diagram(X, D, ax=None, show_grid=False, show_rank_correlation=True):
    """
    Draw a Shepard diagram of input dissimilarities vs map distances.

    Parameters:
    X (np.ndarray): Configuration of objects on the map, shape (n_samples, n_dims).
    D (np.ndarray): Dissimilarity matrix, shape (n_samples, n_samples).
    ax (matplotlib.axes.Axes, optional): Axes object to draw the diagram on.
    show_grid (bool, optional): Whether to show grid lines on the plot.
    show_rank_correlation (bool, optional): Whether to display the rank correlation coefficient.
    """
    distances = cdist(X, X, metric='euclidean')
    distances_flat = distances[np.tril_indices(len(distances), -1)]
    disparities_flat = D[np.tril_indices(len(D), -1)]

    # Fit isotonic regression to the flattened arrays
    ir = IsotonicRegression()
    disp_hat = ir.fit_transform(X=disparities_flat, y=distances_flat)
    disp_hat = normalize_dhat(disp_hat, X.shape[0])

    # Prepare data for plotting
    df = pd.DataFrame({
        'Disparities': disparities_flat,
        'Distances': distances_flat,
        'Fitted Distances': disp_hat
    }).sort_values('Disparities')

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    
    # Plotting the original and fitted distances
    ax.scatter(df['Disparities'], df['Distances'], color="blue", label="Original", alpha=0.6)
    ax.plot(df['Disparities'], df['Fitted Distances'], color="orange", label="Fitted", marker='o', linestyle='-')
    
    ax.set_xlabel('Input Dissimilarity', fontsize=16)
    ax.set_ylabel('Map Distance', fontsize=16)
    ax.legend()

    if show_grid:
        ax.grid(True)

    # Display Spearman rank correlation if requested
    if show_rank_correlation:
        rank_corr = spearmanr(df['Disparities'], df['Distances'])[0]
        ax.text(0.5, -0.15, f'Rank Correlation: {rank_corr:.2f}', transform=ax.transAxes, 
                ha='center', fontsize=14)

    ax.set_ylim(0, df['Distances'].max() * 1.15)
    ax.set_xlim(df['Disparities'].min() * 0.99, df['Disparities'].max() * 1.01)
    
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))

    return ax

def validate_inputs(X_t, color_t, incl_t, n_samples):
    """ Validate the shape and consistency of inputs. """
    if np.any([X.shape != X_t[0].shape for X in X_t]):
        raise ValueError("All input arrays need to be of similar shape.")
    if color_t is not None and any(color.shape[0] != n_samples for color in color_t):
        raise ValueError("Misshaped class arrays.")
    if incl_t is not None and any(incl.shape[0] != n_samples for incl in incl_t):
        raise ValueError("Misshaped inclusion arrays.")

def prepare_transparencies(n_periods, start, end, final):
    """ Prepare transparency values for the plotting periods. """
    return np.linspace(start, end, n_periods-1).tolist() + [final]

def setup_figure(ax, fig_size):
    """ Setup or create figure and axes for plotting. """
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
        return fig, ax, True
    return None, ax, False

def plot_period_data(ax, X, incl, transparencies, draw_map_kws, plot_args):
    """ Plot data for each period. """
    draw_map_kws.update({
        'X': X, 
        'inclusions': incl,
        'scatter_kws': {'alpha': transparencies},
        **plot_args  # Use spread operator to include all additional arguments
    })
    draw_map(**draw_map_kws)

def draw_dynamic_map(X_t, color_t=None, size_t=None, incl_t=None, show_arrows=False, 
                     show_last_positions_only=False, time_labels=None, transparency_start=0.4, 
                     transparency_end=0.8, transparency_final=1., **kwargs):
    """
    Visualizes dynamic map data over multiple periods with options to show movement paths and adjust visual features.

    Parameters
    ----------
    X_t : list of ndarray
        List of arrays containing coordinates for each period, where each array is of shape (n_samples, n_features).
    color_t : list of ndarray, optional
        List of arrays containing color or group identifiers for each period.
    size_t : list of ndarray, optional
        List of arrays containing sizes for each data point in each period.
    incl_t : list of ndarray, optional
        List of arrays indicating if a point should be included in the plot for each period.
    show_arrows : bool, optional
        If True, display arrows showing movement between periods. Default is False.
    show_last_positions_only : bool, optional
        If True, only the last period's positions are shown with arrows indicating the movements from prior periods. Default is False.
    time_labels : list of str, optional
        Labels for each period, displayed in plot annotations or titles.
    transparency_start : float, optional
        Starting transparency level for the first period in the dynamic map.
    transparency_end : float, optional
        Ending transparency level just before the last period in the dynamic map.
    transparency_final : float, optional
        Transparency level for the last period in the dynamic map.
    **kwargs : dict
        Additional keyword arguments to pass to the plotting function or for configuring plot aspects.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the dynamic map, only if not plotted on an existing axis.
    """

    n_periods = len(X_t)
    n_samples = X_t[0].shape[0]
    validate_inputs(X_t, color_t, incl_t, n_samples)

    incl_t = [np.repeat(1, n_samples)]*n_periods if incl_t is None else incl_t
    transparencies = prepare_transparencies(n_periods, transparency_start, transparency_end, transparency_final)

    fig, ax, return_fig = setup_figure(kwargs.get('ax'), kwargs.get('fig_size', (5,5)))
    draw_map_kws = {k: v for k, v in kwargs.items() if k in draw_map.__code__.co_varnames}

    for t in range(n_periods):
        if not show_last_positions_only or t == n_periods - 1:
            plot_args = {'title': None, 'highlighted_labels': None} if t < n_periods - 1 else kwargs
            plot_period_data(ax, X_t[t], incl_t[t], transparencies[t], draw_map_kws, plot_args)

        if show_arrows and t > 0:
            plot_movement_paths(ax, X_t[t-1], X_t[t], incl_t[t-1], incl_t[t], transparencies[t-1])

    style_axes(ax, **{k: kwargs.get(k, True) for k in ['show_axes', 'show_box', 'show_grid', 'axes_at_origin']})

    if 'filename' in kwargs:
        plt.savefig(kwargs['filename'], dpi = 300, format = "png")
    plt.close()

    if return_fig:
        return fig

def plot_movement_paths(ax, X_prev, X_curr, incl_prev, incl_curr, alpha):
    """ Plot movement paths between periods. """
    for i in range(len(X_curr)):
        if incl_curr[i] and incl_prev[i]:
            start_point = X_prev[i]
            end_point = X_curr[i]
            if np.any(start_point != end_point):
                ax.arrow(*start_point, *(end_point - start_point), color='grey', alpha=alpha, linestyle='-', linewidth=1)


def draw_trajectories(Y_ts, labels, selected_labels = None, title = None, 
    show_axes = False, show_box = True, show_grid = False, axes_at_origin = False,
    annotate_periods = True, period_labels = None, ax = None, fig_size = None):
    """ Draw the trajectories of selected objects.

    Parameters
    ----------
    Y_ts : list of ndarrays, each of shape (n_samples, d)
        Sequence of map coordinates.
    labels : ndarray of shape (n_samples,)
        Object labels (str)
    selected_labels : ndarray of shape (n_selected,), optional
        Selected object labels (str), by default None
    title : str, optional
        Figure title, by default None
    annotate_periods : bool, optional
        If true, labels for each period are shown next to each pair of map 
        coordinates, by default True
    period_labels : ndarray of shape (n_periods,), optional
        Period labels (str), by default None
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes for the plot, by default None
    figsize : tuple, optional
        Figure size, by default (12,12)
    """

    n_periods = len(Y_ts)
    n_firms = Y_ts[0].shape[0]
    if selected_labels == None:
        selected_labels = labels

    # If not ax is provided, return the whole FIgure. Else, only draw the plot on the provided axes
    if ax is None:
        return_fig = True
        if fig_size is None:
            fig_size = (5,5)
        fig, ax = plt.subplots(figsize = fig_size)
    else:
        return_fig = False  

    annotations = []

    if period_labels is None and annotate_periods == True:
        period_labels = ["Period " + str(t+1) for t in range(n_periods)]

    for i in range(n_firms):
        if not labels[i] in selected_labels:
            continue
        xs = []
        ys = []
        # Plot the points
        for t in range(n_periods):
            alpha = 1 - (n_periods - t) / n_periods
            alpha = alpha * .5
            x = Y_ts[t][i,0]
            y = Y_ts[t][i,1]
            c = 'black'
            c_line = 'grey'
            label = labels[i]
            ax.scatter(x,y , c = c, alpha = alpha)
            xs.append(x)
            ys.append(y)
            if t == n_periods - 1:
                label = ax.text(x ,y , label, c = c, alpha = .7, fontsize = DEFAULT_FONT_SIZE)
                annotations.append(label)

            elif annotate_periods:
                label = ax.text(x ,y , period_labels[t], c = c_line, alpha = .5, fontsize = DEFAULT_FONT_SIZE * 0.8)
#               texts.append(label)

        # Plot the trajectory
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.plot(xs, ys, c = c_line, alpha = .4)

    style_axes(ax = ax, show_axes= show_axes, show_box = show_box, show_grid = show_grid, axes_at_origin = axes_at_origin)

    if not title is None:
        ax.set_title(title, fontdict = title_fontdict)

    plt.close()
    if return_fig:
        return fig
