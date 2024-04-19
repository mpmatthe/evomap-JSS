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
        "axes.edgecolor": "black",
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
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

def style_axes(ax, show_axes=True, show_box=True, show_grid=False, axes_at_origin=False,
               xlim=(-1, 1), ylim=(-1, 1)):
    """
    Style the axes of a plot with options to show or hide grid, box, and axes, set axes limits,
    and align axes at origin.
    """
    ax.set_ylabel("Dimension 2", fontdict=axis_label_fontdict)
    ax.set_xlabel("Dimension 1", fontdict=axis_label_fontdict)

    ax.grid(show_grid)

    # Set limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if axes_at_origin:
        # Set the axes to origin
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

    if not show_axes:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    else:
        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(True)

    ax.set_frame_on(show_box)

def draw_map(X, label=None, color=None, size=None, inclusions=None, zoom_on_cluster=None, highlighted_labels=None, 
             show_box=True, show_grid=False, show_axes=False, axes_at_origin=False, show_legend=False,
             cmap=None, filename=None, ax=None, fig_size=None, 
             title=None, rotate_labels=0, scatter_kws={}, fontdict=None, rcparams=None):

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
                ax.arrow(*start_point, *(end_point - start_point), color='grey', alpha=alpha, linestyle='--', linewidth=1)


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
