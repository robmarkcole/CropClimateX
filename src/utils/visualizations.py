import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import warnings
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torchmetrics
import einops
import seaborn as sns
import torch
import geopandas as gpd
import xarray as xr

def contrast_stretch(x, lower_percent=1, upper_percent=99, lower=None, upper=None):
    """
    Perform contrast stretching on an image x.
    :param x: numpy array, the image x to be stretched.
    :param lower_percent: lower percentile for stretch.
    :param upper_percent: upper percentile for stretch.
    :return: stretched image x.
    """
    if x.ndim > 2:
        # do it per band (band assumed to be dim 0)
        return np.stack([contrast_stretch(b, lower_percent, upper_percent) for b in x])
    if not lower:
        lower = np.nanpercentile(x, lower_percent)
    if not upper:
        upper = np.nanpercentile(x, upper_percent)
    stretched_band = np.clip((x - lower) / (upper - lower), 0, 1)

    return stretched_band

def contrast_stretch_xr(x, lower_percent=1, upper_percent=99, lower=None, upper=None):
    """
    Perform contrast stretching on an xarray.DataArray.
    :param x: xarray.DataArray, the image data to be stretched.
    :param lower_percent: lower percentile for stretch (default is 1%).
    :param upper_percent: upper percentile for stretch (default is 99%).
    :param lower: explicit lower limit for stretch. Overrides lower_percent if provided.
    :param upper: explicit upper limit for stretch. Overrides upper_percent if provided.
    :return: xarray.DataArray with contrast-stretched values.
    """
    def stretch_band(band):
        nonlocal lower, upper
        if lower is None:
            lower = np.nanpercentile(band, lower_percent)
        if upper is None:
            upper = np.nanpercentile(band, upper_percent)
        return np.clip((band - lower) / (upper - lower), 0, 1)

    if 'variable' in x.sizes and x.sizes['variable'] > 1:
        stretched = xr.concat([stretch_band(x.isel(variable=band)) for band in range(x.sizes['variable'])], dim='variable')
    else:
        stretched = stretch_band(x)

    return stretched

def plot_img_sequence(y=None, pred=None, x=None, mask=None, cmap=None, plot_interval=5, subtitles=None, **kwargs):
    """Plot sequences of images of the predictions, ground truth, input and mask."""
    vals = [v for v in [y, pred, x, mask] if v is not None]
    if len(vals) == 0:
        raise ValueError("No input data found.")
    length = len(vals[0])
    if not all(len(v) == len(vals[0]) for v in vals[1:]):
        # raise ValueError("Number of images does not match.")
        # assume that one variable is missing the first time steps and remove them
        length = min([len(v) for v in vals])
        y, pred, x, mask = [v[-length:] if v is not None else None for v in [y, pred, x, mask]]
        warnings.warn(f"Number of subplots does not match the number of images. Only last {length} images are used.", RuntimeWarning)

    nr_subplots = length // plot_interval
    nr_subplots += 1 if length % plot_interval else 0
    nr_plots = 2
    nr_plots += 1 if not x is None else 0
    nr_plots += 1 if not mask is None else 0
    fig = plt.figure(constrained_layout=True, figsize=(nr_plots*5, nr_subplots*4))
    subfigs = fig.subfigures(nr_subplots)
    if nr_subplots < 2:
        subfigs = [subfigs]
    for i,j in zip(range(nr_subplots), range(0, length, plot_interval)):
        plot_image(y[j] if y is not None else None, pred[j] if pred is not None else None, x=x[i] if x is not None else None, mask=mask[j] if mask is not None else None, cmap=cmap, parent_fig=subfigs[i], **kwargs)
        if subtitles is not None:
            subfigs[i].suptitle(subtitles[j])
        else:
            subfigs[i].suptitle(f'Step {j}')

    return fig

def plot_conditional_sequence(x,y,preds,x_aux,y_aux,m, cmap=None, parent_fig=None, contrast_stretch_percentile=False, apply_mask=False):
    if apply_mask:
        # repeat in channel dimension
        m = m.repeat(1,pred.shape[1],1,1)
        preds[m.bool()] = 1
        # preds = preds*m
        # preds = torch.where(preds==0, torch.ones_like(preds), preds)

    if x.shape[-3] > 3:
        # take first three channels
        x = x[..., :3, :, :]
    if y.shape[-3] > 3:
        y = y[..., :3, :, :]
    if preds.shape[-3] > 3:
        preds = preds[..., :3, :, :]


    row_titles = ["GT", "Pred", "Mask", "Aux"]
    col_titles = ["Context", "Target"]  # Modify if more than 2 time steps
    cols = len(x)+len(y)
    rows = 3
    # Create figure and axes
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows),
                            gridspec_kw={'wspace': 0, 'hspace': 0})

    # Plot images
    for i in range(rows):
        # concat input in time
        if i == 0:
            img = np.concatenate([x,y], axis=0) 
        elif i == 1:
            img = np.concatenate([np.ones_like(x),preds], axis=0)
        elif i == 2:
            shape = list(x.shape)
            shape[-3] = 1
            img = np.concatenate([np.zeros(shape),1-m], axis=0)
            cmap = 'gray'
        elif i == 3:
            img = np.concatenate([x_aux,y_aux], axis=0)
        for j in range(cols):
            ax = axes[i, j]
            plot_img(img[j], ax, cmap=cmap, contrast_stretch_percentile=contrast_stretch_percentile)

            # Set column titles on the first row
            if i == 0:
                if j == 0:
                    ax.set_title(col_titles[0], fontsize=12, fontweight='bold')
                elif j == len(x):
                    ax.set_title(col_titles[1], fontsize=12, fontweight='bold')
            if j == len(x)-1:
                ax.spines['right'].set_color('yellow')
                ax.spines['right'].set_linewidth(3)

    # Add row titles to the first column
    for i in range(rows):
        axes[i, 0].set_ylabel(row_titles[i], fontsize=12, fontweight='bold', rotation=90, labelpad=10)

    return fig

def plot_img(x, ax, cmap=None, contrast_stretch_percentile=False):
    """Plot a single image."""
    x = x.squeeze()
    if x.ndim < 3 and cmap is None:
        cmap = 'RdYlGn'
    if x.ndim == 3 and x.shape[0] < x.shape[1]:
        x = np.moveaxis(x, 0, -1)
    if contrast_stretch_percentile:
        x = contrast_stretch(x, *contrast_stretch_percentile)
    if x.min() < 0:
        # assume -1 to 1 range -> shift to 0 to 1
        x = (x + 1) / 2

    ax.imshow(x, cmap=cmap, aspect='auto')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(True)
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')

def plot_image(y=None, pred=None, x=None, mask=None, cmap=None, parent_fig=None, contrast_stretch_percentile=False):
    """Plot the input, ground truth, prediction, residuals and mask if they are provided.
    :param y: numpy array, ground truth.
    :param pred: numpy array, prediction.
    :param x: numpy array, input.
    :param mask: numpy array, mask.
    :param cmap: str, colormap to use.
    :param parent_fig: matplotlib figure, parent figure to plot on.
    :param contrast_stretch_percentile: (optional) list, lower and upper percentile for contrast stretching."""
    title = []
    if x is not None:
        title.append('Input')
    if y is not None:
        title.append('Ground Truth')
    if pred is not None:
        title.append('Prediction')
    if y is not None and pred is not None:
        title.append('Residuals')
    if mask is not None:
        mask = mask.squeeze()
        title.append('Mask')
    nr_plots = len(title)

    if parent_fig is None:
        fig = plt.figure(figsize=(nr_plots*5,4))
    else:
        fig = parent_fig
    ax = fig.subplots(1,nr_plots)

    if not isinstance(ax, np.ndarray):
        ax = [ax]
    
    nr = 0
    # input
    if x is not None:
        x = x.squeeze()
        if contrast_stretch_percentile:
            x = contrast_stretch(x, *contrast_stretch_percentile)
        # bring channels to last dimension if they are first
        if x.ndim == 3 and x.shape[0] < x.shape[1]:
            x = np.moveaxis(x, 0, -1)
        min, max = np.nanmin(x), np.nanmax(x)
        im = ax[nr].imshow(x, cmap=cmap, vmin=min, vmax=max)
        nr += 1

    if x is not None and y is None and pred is None and cmap is not None:
        divider = make_axes_locatable(ax[nr-1])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im, cax=cax)

    if y is not None or pred is not None:

        data = [y, pred]
        mins = maxs = [] # images mins and maxs
        for i in data:
            if i is not None:
                mins.append(np.nanmin(i))
                maxs.append(np.nanmax(i))

        min = np.min(mins)
        max = np.max(maxs)

        for i in range(len(data)):
            if data[i] is not None:
                data[i] = data[i].squeeze()
                if contrast_stretch_percentile:
                    data[i] = contrast_stretch(data[i], *contrast_stretch_percentile, lower=min, upper=max)
                # bring channels to last dimension if they are first
                if data[i].ndim == 3 and data[i].shape[0] < data[i].shape[1]:
                    data[i] = np.moveaxis(data[i], 0, -1)

        # ground truth + prediction
        for i in range(len(data)):
            if data[i] is not None:
                im = ax[nr].imshow(data[i], vmin=min, vmax=max, cmap=cmap)
                nr += 1

        # colorbar
        divider = make_axes_locatable(ax[nr-1])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im, cax=cax)
    
        if y is not None and pred is not None: # residuals
            divnorm=matplotlib.colors.TwoSlopeNorm(vcenter=0.)
            residuals = data[0]-data[1]
            im = ax[nr].imshow(residuals, cmap='bwr', norm=divnorm)
            nr += 1
            divider = make_axes_locatable(ax[nr-1])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(im, cax=cax)

    if mask is not None:
        ax[-1].imshow(mask, cmap='gray', vmin=0, vmax=1)
    
    for i in range(nr_plots):
        ax[i].set_aspect('equal', 'box')
        ax[i].get_yaxis().set_visible(False)
        ax[i].get_xaxis().set_visible(False)
        ax[i].set_title(title[i])

    return fig

def plot_residuals(y, pred, **kwargs):
    if y.ndim > 1:
        y = y.flatten()
    if pred.ndim > 1:
        pred = pred.flatten()
    residuals = y - pred
    fig, ax = plt.subplots(figsize=(10, 5))
    # zero line
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.scatter(pred, residuals, s=1)
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Residuals')
    return fig

def plot_y_pred(y, pred):
    if y.ndim > 1:
        y = y.flatten()
    if pred.ndim > 1:
        pred = pred.flatten()
    fig, ax = plt.subplots(figsize=(10, 5))
    # 45 degree line
    ax.plot([min(pred), max(pred)], [min(y), max(y)], transform=ax.transAxes, color='black', linestyle='--', alpha=0.5)
    ax.scatter(y, pred, s=1)
    ax.set_xlabel('Ground Truth')
    ax.set_ylabel('Prediction')
    return fig

def plot_confusion_matrix(y, preds, handle_preds=False, **kwargs):
    if y.ndim > 1:
        y = y.flatten()
    if preds.ndim > 2:
        # class dimension assumed to be on the second axis (see wrapper)
        # -> move it to last axis
        preds = einops.rearrange(preds, 'b c ... -> (b ...) c')
        # flatten without last axis
        preds = preds.flatten(end_dim=-2)
    else:
        preds = preds.flatten()

    if handle_preds == 'round':
        # preds assumed to be regression values -> convert to int by rounding
        preds = torch.round(preds).long()
        preds = preds.flatten() # since y is flattened

    # create and plot matrix
    cm = torchmetrics.ConfusionMatrix(**kwargs)(preds, y)
    cm = cm.round(decimals=2)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', annot_kws={"fontsize":14})
    # font size labels
    ax.tick_params(labelsize=14)
    # font size colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    return fig

def plot_histogram(*args, **kwargs):
    fig, ax = plt.subplots(len(args), figsize=(10, len(args)*5))
    if len(args) == 1:
        ax = [ax]
    for i, data in enumerate(args):
        ax[i].hist(data, **kwargs)
    fig.tight_layout()
    return fig

def plot_county_residuals(y, preds, meta, fn_counties, plot_lines='states', bounds='auto', **kwargs):
    res = preds - y
    geoids = [m['geoid'] for m in meta]
    # mean of res by geoid
    u_geoids = set(geoids)
    res_by_geoid = {g: [] for g in u_geoids}
    for i, g in enumerate(geoids):
        res_by_geoid[g].append(res[i])
    mean_res_by_geoid = {g: np.mean(res_by_geoid[g]) for g in u_geoids}

    gdf = gpd.read_file(fn_counties)
    gdf_res = gdf[gdf['GEOID'].isin(geoids)]
    gdf_res.set_index('GEOID', inplace=True)

    # add the mean residuals to the geodataframe
    gdf_res['mean_residual'] = [mean_res_by_geoid[g] for g in gdf_res.index]
    
    # plot the residuals
    if bounds == 'auto':
        # bounds = np.linspace(torch.min(res), torch.max(res), 11)
        vmin, vmax = torch.min(res), torch.max(res)
        neg_boundaries = np.linspace(vmin, 0, num=5, endpoint=False)
        pos_boundaries = np.linspace(0, vmax, num=6)

        # Combine boundaries, making sure 0 is included
        bounds = np.concatenate([neg_boundaries, pos_boundaries])
    else:
        bounds = np.array(bounds)    
    cmap = plt.cm.RdBu
    norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N, extend='both')
    
    fig, ax = plt.subplots(figsize=(20, 10))

    gdf_res.plot(column='mean_residual', ax=ax, legend=True, norm=norm, cmap=cmap, edgecolor='black', linewidth=0.4, legend_kwds={"shrink":.8})
    if plot_lines == 'counties':
        gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5, alpha=0.5)
    elif plot_lines == 'states':
        url = "https://www2.census.gov/geo/tiger/TIGER2024/STATE/tl_2024_us_state.zip"
        states = gpd.read_file(url)
        # only mainland
        states = states[~states['NAME'].isin(["Puerto Rico", "Guam", "United States Virgin Islands", "American Samoa", "Northern Mariana Islands", "Alaska", "Hawaii", "Commonwealth of the Northern Mariana Islands"])]
        states.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)
    else:
        raise ValueError("Invalid value for plot_lines. Choose 'counties' or 'states'.")

    ax.axis('off')
    return fig
