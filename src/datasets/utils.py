import math
import warnings
import xarray as xr
import numpy as np
import re
from scipy import stats
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DistributedSampler
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
from typing import Iterator, List, Optional, Union
from operator import itemgetter

def get_length_from_ratio(dataset_length, split_ratio=(.7,.2,.1)):
    """Get the length of each train, val and test set from the split_ratio and the dataset length."""
    if sum(split_ratio) > 3: # assume that this are already the lengths
        warnings.warn('split_ratio is greater than 3, assuming that this are the number of samples for each set. Returning split_ratio as is.')
        return split_ratio
    if not math.isclose(sum(split_ratio), 1):
        raise ValueError('split_ratio should sum to 1, got {} = {}'.format(split_ratio, sum(split_ratio)))
    lens = [math.floor(dataset_length*i) for i in split_ratio]
    remain = dataset_length - sum(lens)
    for i in range(1,remain+1):
        # if they have 0 samples add it, or per round robin
        if lens[1] < 1:
            lens[1] += 1
        elif lens[2] < 1 and split_ratio[2] > 0:
            lens[2] += 1
        else:
            idx = i % 3
            lens[idx] += 1
    return lens

bands_freq = {
    'modis': '8D',
    'landsat': '16D',
    'sen2': '15D',
    'lai': '8D',
    'lst': '8D',
    'daymet': '1D',
    'usdm': '7D',   
}

def gap_filling_yearly(ds, freq:str, start:str, end:str):
    if 'time' not in ds.coords:
        return ds
    # each year starts with 01-01, split data to years
    gs = ds.groupby('time.year')
    if len(gs) == 1:
        ds = gap_filling(ds, freq, start, end)
    else:
        dss = []
        for g,v in gs:
            g = str(g)
            # set new start and end, if the year is not complete
            if end[:4] == g:
                e = end
            else:
                e = f"{g}-12-31"
            if start[:4] == g:
                s = start
            else:
                s = f"{g}-01-01"
            dss.append(gap_filling(v, freq, s, e))
        ds = xr.concat(dss, dim='time').sortby('time')
    return ds

def gap_filling(ds, freq:str, start:str, end:str):
    if 'time' not in ds.coords:
        return ds
    # round data to days (otherwise it will not match if there are hours in the time steps)
    ds['time'] = ds['time'].dt.floor('D')
    i = 1
    while True:
        # find first time step in series
        if pd.to_datetime(start) + i * pd.Timedelta(freq) > ds.time[0]:
            # compute the first missing time step
            start = (ds.time[0].values - (i-1) * pd.Timedelta(freq)).strftime('%Y-%m-%d')
            break
        i += 1
    times = pd.date_range(start=start, end=end, freq=freq)
    # rm last day in leap years
    if freq == '1D' or freq == 'D':
        times = times[~((times.month == 12) & 
                (times.day == 31) & 
                (times.is_leap_year))]
    # fill time steps with nans
    ds = ds.reindex(time=times)
    return ds

def resample_yearly(ds, dict, master=None, repeat_in_time=True):
    if 'time' not in ds.coords:
        # fall back to normal resampling
        return resample(ds, dict, master, repeat_in_time)
    # each year starts with 01-01, split data to years
    gs = ds.groupby('time.year')
    gm = master.groupby('time.year') if master is not None else None
    if len(gs) == 1:
        ds = resample(ds, dict, master, repeat_in_time)
    else:
        dss = []
        for s,m in zip(gs, gm) if gm is not None else zip(gs, [None]*len(gs)):
            s = s[1]
            m = m[1] if gm is not None else None
            dss.append(resample(s, dict, m, repeat_in_time))
        ds = xr.concat(dss, dim='time').sortby('time')
    return ds

def resample(ds, dict, master=None, repeat_in_time=True):
    """spatially and temporally resample the an xarray dataset according to the configuration dict.
    method should contain first interpolation method and second the reduction method. (fill only possible for temporal resampling)
    example:
        {'spatial': {'method':'linear/mean', 'size':200},
         'temporal': {'method':'linear/mean', 'size':'2D'}}
        {'spatial': {'method':'linear/mean', 'size': 'master'},
         'temporal': {'method':'linear/mean', 'size':'2D'}}

    """
    if dict:
        if isinstance(ds, xr.Dataset):
            dtypes = list(set(ds[var].dtype for var in ds.data_vars))
            # convert bool to int for mode/interp
            for var in ds.data_vars:
                if np.issubdtype(ds[var].dtype, np.bool_):
                    ds[var] = ds[var].astype('int')
        else:
            dtypes = [ds.dtype]
            # convert bool to int for mode/interp
            if np.issubdtype(ds.dtype, np.bool_):
                ds = ds.astype('int')

        if len(dtypes) > 1 and not (all(np.issubdtype(dtype, np.floating) for dtype in dtypes) or all(np.issubdtype(dtype, np.integer) for dtype in dtypes)):
            warnings.warn(f'Resampling of mixed datatypes is not supported, found {dtypes}.')
        if repeat_in_time and (not 'time' in ds.coords or ds.coords['time'].size <= 1):
            # repeat the static data as dynamic
            if master is not None:
                if 'time' in ds.dims:
                    # drop time dimension
                    ds = ds.squeeze('time')
                    del ds.coords['time']
                if not 'time' in ds.dims:
                    # give it masters time dimension
                    ds_expand = [ds.expand_dims(time=[t]) for t in master.time.values]
                    ds = xr.concat(ds_expand, dim='time')
        else:
            if dict.get('temporal', None):
                ds['time'] = ds['time'].dt.floor('D')
                if master is not None:
                    master['time'] = master['time'].dt.floor('D')
                temporal_dict = dict['temporal']
                if len(set(ds.time.dt.year.values)) > 1 and 'master' not in temporal_dict['size']:
                    raise ValueError('Temporal resampling is only possible within a year.') 
                if 'fill' in temporal_dict['method']:
                    # fill time steps with nans
                    times = pd.date_range(start=str(ds.time[0].values), end=str(ds.time[-1].values), freq=temporal_dict['size'])
                    ds = ds.reindex(time=times)
                else:
                    if temporal_dict['size'] == 'master':
                        if master is not None:
                            # always temporally interpolate to the master time
                            ds = ds.interp(time=master.time, method=temporal_dict['method'].split('/')[0], kwargs={"fill_value": "extrapolate"})
                    else:
                        nr = int(re.findall(r'\d+', temporal_dict['size'])[0])
                        format = re.findall(r'[a-zA-Z]+', temporal_dict['size'])[0]
                        size_delta = np.timedelta64(nr, format)
                        # get ds time resolution
                        time_resolution = (ds.time[1] - ds.time[0]).astype('timedelta64[ns]')
                        if time_resolution < size_delta:
                            # downsample
                            if 'mode' in temporal_dict['method']:
                                ds = ds.resample(time=temporal_dict['size']).reduce(mode_reduction)
                            elif 'mean' in temporal_dict['method']:
                                ds = ds.resample(time=temporal_dict['size']).mean()
                            else:
                                raise ValueError(f'Unknown method {temporal_dict["method"]} for temporal resampling.')
                        elif time_resolution > size_delta:
                            # upsample
                            ds = ds.resample(time=temporal_dict['size']).interpolate(temporal_dict['method'].split('/')[0])
    
        if dict.get('spatial', None):
            spatial_dict = dict['spatial']
            if spatial_dict['size'] == 'master' and master:
                size_x = master.sizes['x']
                size_y = master.sizes['y']
                scale_x = ds.sizes['x'] / master.sizes['x']
                scale_y = ds.sizes['y'] / master.sizes['y']
            elif spatial_dict['size'] != 'master':
                size_x = spatial_dict['size']
                size_y = spatial_dict['size']
                scale_x = ds.sizes['x'] / spatial_dict['size']
                scale_y = ds.sizes['y'] / spatial_dict['size']
            else:
                # skip spatial resampling -> assumed it is master (first data source) and should stay like it is
                return ds
            assert (scale_x > 1 and scale_y > 1) or (scale_x <= 1 and scale_y <= 1), 'The spatial resampling contains up- and downsampling.'
            if scale_x > 1:
                # downscale
                if scale_x != int(scale_x) or scale_y != int(scale_y):
                    # remove edge pixels if not divisible
                    x_new_size = size_x * int(scale_x)
                    y_new_size = size_y * int(scale_y)
                    x_trim_start = (ds.sizes['x'] - x_new_size) // 2
                    x_trim_end = x_trim_start + x_new_size
                    y_trim_start = (ds.sizes['y'] - y_new_size) // 2
                    y_trim_end = y_trim_start + y_new_size
                    ds = ds.isel(x=slice(int(x_trim_start), int(x_trim_end)), y=slice(int(y_trim_start), int(y_trim_end)))
                if 'mode' in spatial_dict['method']:
                    ds = ds.coarsen({'x': int(scale_x), 'y': int(scale_y)}, boundary='trim').reduce(mode_reduction)
                elif 'mean' in spatial_dict['method']:
                    ds = ds.coarsen({'x': int(scale_x), 'y': int(scale_y)}, boundary='trim').mean()
                else:
                    raise ValueError(f'Unknown method {spatial_dict["method"]} for spatial resampling.')
            elif scale_x < 1:
                # upscale
                if master is not None and 'master' == spatial_dict['size']:
                    x = resample_coords(ds.coords['x'], master.sizes['x'])
                    y = resample_coords(ds.coords['y'], master.sizes['y'])
                else:
                    x = resample_coords(ds.coords['x'], spatial_dict['size'])
                    y = resample_coords(ds.coords['y'], spatial_dict['size'])
                ds = ds.interp({'x':x, 'y':y}, method=spatial_dict['method'].split('/')[0])
    return ds

def resample_coords(coords, size):
    """Resample the coordinates to a new size."""
    num = int(size)
    assert size == float(num), 'The size should be an integer.'
    min, max = coords.min().values, coords.max().values
    new_coords = np.linspace(min, max, num=num, endpoint=True)
    # if order needs to be reverted
    if coords[0] > coords[-1]:
        new_coords = np.flip(new_coords)
    return new_coords

def mode_reduction(block, axis):
    mode, _ = stats.mode(block, axis=axis)
    return mode

def number_to_range(input_num, divider):
    """transform the input number to a range of the divider."""
    start = (input_num // divider) * divider
    end = start + divider - 1
    return f"{start}-{end}"

def make_weights_for_balanced_classes(ds, n_classes, batch_size, target_name, num_workers=8, fn_classes=None):
    """
    Make a weight tensor for a dataset with balanced classes.
    Adapted from https://gist.github.com/simonmoesorensen/ac590e8e25ac8b1c322519d2d8c73676
    """
    count = torch.zeros(n_classes)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    n_samples = 0
    for batch in tqdm(dl, desc="Counting classes"):
        y = batch[target_name]
        y = fn_classes(y) if fn_classes else y
        idx, counts = y.unique(return_counts=True)
        count[idx] += counts
        n_samples += y.shape[0] if y.dim() > 0 else len(y)

    N = count.sum()
    weight_per_class = N / count

    weight = torch.zeros(n_samples)

    for i, batch in tqdm(enumerate(dl), desc="Apply weights", total=len(dl)):
        x, y = batch['x'], batch['y']
        idx = torch.arange(0, x.shape[0]) + (i * batch_size)
        idx = idx.to(dtype=torch.long)
        weights = weight_per_class[y]
        if weights.dim() > 1: # in case of multiple targets, take avg of weights
            weights = torch.mean(weights, dim=tuple(range(1,weights.dim())))
        weight[idx] = weights

    return weight


class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.

    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.

        Args:
            index: index of the element in the dataset

        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)

class DistributedSamplerWrapper(DistributedSampler):
    # copied from here: https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/sampler.py#L499
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.

    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.

    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """

        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.

        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))