from typing import Any, Dict, Optional, Tuple
from torch.utils.data import Dataset
import torch
import xarray as xr
import geopandas as gpd
import pandas as pd
import os
import numpy as np
from src.datasets.utils import resample_yearly, number_to_range, bands_freq, gap_filling_yearly
import zarr
import itertools
import glob
import math
import re
import warnings
import hashlib

def contains_year(s):
    pattern = r'\b(1[0-9]{3}|2[0-9]{3})\b'
    return bool(re.search(pattern, s))

def create_dates(nr_days, years, bands, rng, nr_samples=None):
    """Create a list of time ranges for the given years and bands.
    The time ranges are created by randomly selecting a start date and then creating ranges with the given frequency.
    rng is a numpy random state to make sure the same time ranges are created for the same geoid and are independend from the general seed."""
    rand_nr = rng.randint(0, int(nr_days[:-1])) # random start day
    # time range is frequency -> make split dynamically
    start = (pd.to_datetime(f'{years[0]}-01-01') + pd.Timedelta(days=rand_nr)).strftime('%Y-%m-%d') # start somewhere in the first period
    days = nr_days #+ pd.Timedelta(bands_freq[next(iter(bands))]) # add one freq step to get a buffer
    # slice through the years until no more time ranges can be created
    date_range_start = list(pd.date_range(start=start, end=f'{years[-1]}-12-31', freq=days).strftime('%Y-%m-%d'))
    date_range_end = list((pd.date_range(start=start, end=f'{years[-1]}-12-31', freq=days, inclusive='right') - pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
    date_range_start[0] = start
    date_range = list(zip(date_range_start, date_range_end))
    # drop if date is not in the years
    date_range = [d for d in date_range if int(d[0][:4]) in years and int(d[1][:4]) in years]
    if nr_samples:
        idxs = rng.choice(len(date_ranges), nr_samples, replace=False)
        date_ranges = [date_ranges[i] for i in idxs]
    return date_range

def create_dates_fixed(nr_days, dates, years, rng, nr_samples=None):
    """Create a list of time ranges in between the dates for the given years and bands."""
    date_ranges = []
    days = int(nr_days[:-1])
    for y in years:
        for d in dates:
            start = pd.to_datetime(f'{y}-{d[0]}')
            end = pd.to_datetime(f'{y}-{d[1]}')
            day_diff = (end - start).days
            if days > day_diff:
                raise ValueError(f'Frequency {days} is bigger than the difference between the start and end date of {day_diff} days')
            rand_nr = rng.randint(0, day_diff-days) # random start day
            start = (start + pd.Timedelta(days=rand_nr)).strftime('%Y-%m-%d')
            date_range_start = list(pd.date_range(start=start, end=end, freq=nr_days).strftime('%Y-%m-%d'))
            date_range_end = list((pd.date_range(start=start, end=end, freq=nr_days, inclusive='right') - pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
            date_range_start[0] = start
            date_range = list(zip(date_range_start, date_range_end))
            date_ranges.extend(date_range)
    if len(date_ranges) == 0:
        raise ValueError(f'No time ranges could be created for the given years {years} and dates {dates}')
    if nr_samples:
        idxs = rng.choice(len(date_ranges), nr_samples, replace=False)
        date_ranges = [date_ranges[i] for i in idxs]
    return date_ranges

class MinicubeDataset(Dataset):
    def __init__(self, data_dir: str, x_transform=None, y_transform=None, bands:dict=None,
                pred_bands:dict=None, time_range:dict={}, years:list=None, geoids:list=None,
                resampling:dict={}, spatial_splitting:int=0, apply_cloud_mask=False, return_cloud_mask=False,
                return_meta_data=False, return_label_data=False, return_y_sequence=True, roll_y_forward=False,
                y_time_range:dict={}, seed=42, handle_nans='none', gap_filling=False) -> None:
        self.data_dir = data_dir
        self.geoids = geoids
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.bands = bands
        self.pred_bands = pred_bands
        self.resampling = resampling
        self.gap_filling = gap_filling
        self.apply_cloud_mask = apply_cloud_mask
        self.return_cloud_mask = return_cloud_mask
        self.return_meta_data = return_meta_data
        self.return_label_data = return_label_data
        self.return_y_sequence = return_y_sequence
        self.roll_y_forward = roll_y_forward
        self.handle_nans = handle_nans

        # create the time ranges as entry in df
        if years is None:
            years = list(range(2018, 2023))
        years = sorted(years)

        rng = np.random.RandomState(seed)

        # flatten dict
        str_time = "-".join(f"{key}-{value}" for key, value in time_range.items())
        y_str_time = "-".join(f"{key}-{value}" for key, value in y_time_range.items())
        
        self.side_spatial_splitting = math.sqrt(spatial_splitting)

        self.nr_time_steps = time_range['nr_time_steps'] if time_range and 'nr_time_steps' in time_range else None
        self.y_nr_time_steps = y_time_range['nr_time_steps'] if time_range and 'nr_time_steps' in y_time_range else None
        
        # create hash for the ids
        data_str = ''.join(sorted(geoids))
        hash_ids = hashlib.sha256(data_str.encode()).hexdigest()
        # save precomputed in file name 
        fn = os.path.join(data_dir, f'labels_{str_time}_{y_str_time}_{"".join([str(y) for y in years])}_{spatial_splitting}_{hash_ids}.csv')
        if os.path.exists(fn):
            self.labels = pd.read_pickle(fn)
            return

        # load the patches
        if not isinstance(geoids, list):
            geoids = list(geoids)
        self.labels = pd.DataFrame({'GEOID_PID': geoids})

        if spatial_splitting > 2:
            # create multiple samples from one
            self.labels = pd.DataFrame({
                'GEOID_PID': self.labels['GEOID_PID'].repeat(spatial_splitting).reset_index(drop=True),
                'split_part': self.labels['GEOID_PID'].apply(lambda x: [i for i in range(spatial_splitting)]).explode().reset_index(drop=True)
            })
        else:
            self.labels['split_part'] = -1

        if time_range:
            nr_samples = None
            if 'nr_samples' in time_range:
                nr_samples = time_range['nr_samples']
            if 'select' in time_range and 'days' in time_range:
                dates = time_range['select']
                self.labels['time_range'] = self.labels['GEOID_PID'].apply(lambda x: [i for i in create_dates_fixed(time_range['days'], dates, years, rng, nr_samples)])
            elif 'days' in time_range:
                # add the time ranges to the labels by repeating the geoids
                # every sample gets a different time range
                self.labels['time_range'] = self.labels['GEOID_PID'].apply(lambda x: [i for i in create_dates(time_range['days'], years, bands, rng, nr_samples)])
            elif 'select' in time_range:
                # time ranges are provided already, e.g. (01-01 , 12-31) -> add years
                if not isinstance(time_range['days'][0], (tuple,list)):
                    time_range['days'] = [time_range['days']]
                if contains_year(time_range['days'][0][0]):
                    date_range = time_range['days']
                else:
                    date_range = [(f'{year}-{i[0]}', f'{year}-{i[1]}') for i in time_range['days'] for year in years]
                self.labels['time_range'] = self.labels['GEOID_PID'].apply(lambda x: [i for i in date_range])
        else:
            warnings.warn(f'time_range not properly defined, taking all time steps: {years[0]}-01-01 - {years[-1]}-12-31')
            self.labels['time_range'] = self.labels['GEOID_PID'].apply(lambda x: [(f'{years[0]}-01-01', f'{years[-1]}-12-31')])
        self.labels = self.labels.explode('time_range').reset_index(drop=True)

        if y_time_range:
            if 'days' in time_range:
                self.labels['y_time_range'] = self.labels['time_range'].apply(lambda x: ((pd.to_datetime(x[0]) + pd.Timedelta(y_time_range['days'])).strftime('%Y-%m-%d'), (pd.to_datetime(x[1]) + pd.Timedelta(y_time_range['days'])).strftime('%Y-%m-%d')))
            elif 'select' in time_range:
                # time ranges are provided already, e.g. (01-01 , 12-31) -> add years
                if not isinstance(time_range['days'][0], (tuple,list)):
                    time_range['days'] = [time_range['days']]
                if contains_year(time_range['days'][0][0]):
                    date_range = time_range['days']
                else:
                    date_range = [(f'{year}-{i[0]}', f'{year}-{i[1]}') for i in time_range['days'] for year in years]
                self.labels['y_time_range'] = self.labels['GEOID_PID'].apply(lambda x: [i for i in date_range])
            else:
                warnings.warn('y_time_range not properly defined')
        
        self.labels.to_pickle(fn)

    def __len__(self) -> int:
        return len(self.labels.index)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        geoid = self.labels.iloc[idx]['GEOID_PID'].split('_')[0] # read the geoid
        pid = self.labels.iloc[idx]['GEOID_PID'].split('_')[1] # read the patch id
        split_part = self.labels.iloc[idx]['split_part'] # read which part of the sample to load
        time_range = self.labels.iloc[idx]['time_range'] # read the time and to filter by it
        if 'y_time_range' in self.labels.columns:
            y_time_range = self.labels.iloc[idx]['y_time_range']
            y_nr_time_steps = self.y_nr_time_steps
        else:
            y_time_range = time_range
            y_nr_time_steps = self.nr_time_steps

        x, master = self.load(self.bands, geoid, pid, time_range, self.nr_time_steps, split_part)
        
        if self.x_transform:
            x = self.x_transform(x)
        
        if self.handle_nans == 'zeros':
            x = torch.where(torch.isnan(x), 0, x)

        out = {'x':x}

        if self.pred_bands:
            y, _ = self.load(self.pred_bands, geoid, pid, y_time_range, y_nr_time_steps, split_part, master, y=True)

            if self.roll_y_forward:
                # roll y forward by one step
                y = y.roll(-1, dims=0)
            if not self.return_y_sequence:
                # return last step
                y = y[-1]

            if self.y_transform:
                y = self.y_transform(y)
        
        if self.pred_bands:
            if self.handle_nans == 'zeros':
                y = torch.where(torch.isnan(y), 0, y)
            out['y'] = y

        modality = list(self.pred_bands.keys())[0] if self.pred_bands else list(self.bands.keys())[0]
        if self.apply_cloud_mask:
            # replace cloud values by nan
            m, _ = self.load({modality: ['c_mask']}, geoid, pid, y_time_range, y_nr_time_steps, split_part, master)
            out['x'] = torch.where(m==1, np.nan, x)
            out['y'] = torch.where(m==1, np.nan, y)

        if self.return_cloud_mask:
            m, _ = self.load({modality: ['c_mask']}, geoid, pid, y_time_range, y_nr_time_steps, split_part, master, y=True)
            m = 1 - m # flip the mask
            out['mask'] = m
        if self.return_meta_data:
            out['meta'] = {'time':master.time.values.astype('int64'), 'geoid':geoid, 'pid':pid, 'split_part':split_part}
        if self.return_label_data:
            out['label'] = {'geoid': geoid, 'pid':pid, 'split_part':split_part, 'time': f"{time_range[0]}-{time_range[-1]}"}
        
        return out

    def load(self, bands, geoid, pid, time, nr_time_steps, split_part, master=None, y=False):
        x = None
        nr_band = 0
        nr_bands = sum([len(bands[m]) for m in bands])
        for m in bands:
            # load the sample
            fn = os.path.join(self.data_dir, m, f'{m}_{geoid}_{number_to_range(int(pid), 10)}.zarr')
            ds = xr.open_zarr(fn, group=pid)
            # select bands
            ds = ds[bands[m]]
            # select time range
            if 'time' in ds.dims:
                ds = ds.sel(time=slice(time[0], time[-1]))
            # select one spatial part of the sample, which is defined by the split_part
            if self.side_spatial_splitting > 1:
                if ds.sizes['x']%self.side_spatial_splitting != 0:
                    raise ValueError(f"spatial_splitting into {self.side_spatial_splitting} parts does not fit the sample size {ds.sizes['x']}")
                length = int(ds.sizes['x']/self.side_spatial_splitting)
                row = split_part // self.side_spatial_splitting
                col = split_part % self.side_spatial_splitting
                ds = ds.isel(x=slice(int(col * length), int((col + 1) * length)), y=slice(int(row * length), int((row + 1) * length)))
            # fill gaps
            if self.gap_filling and m in bands_freq:
                ds = gap_filling_yearly(ds, bands_freq[m], time[0], time[-1])
            # process to diff spatial/temporal resolution
            if self.resampling:
                res_dict = self.resampling
                if 'x' in self.resampling or 'y' in self.resampling:
                    # if there is a specific resampling for x or y, use it
                    if 'x' in self.resampling and not y:
                        res_dict = self.resampling['x']
                    elif 'y' in self.resampling and y:
                        res_dict = self.resampling['y']
                    else:
                        res_dict = {}
                if m in res_dict:
                    ds = resample_yearly(ds, res_dict[m], master)
                elif 'default' in res_dict:
                    ds = resample_yearly(ds, res_dict['default'], master)

            # select only the x first steps or add nans (to make sure all time series have same length)
            if nr_time_steps:
                if 'time' in ds.dims and ds.sizes['time'] > nr_time_steps:
                    ds = ds.isel(time=slice(0, nr_time_steps))
                elif 'time' in ds.dims and ds.sizes['time'] < nr_time_steps:
                    additional_times = pd.date_range(start=ds.time[-1].dt.strftime('%Y-%m-%d').item(), periods=nr_time_steps-ds.sizes['time']+1,
                                                     freq=bands_freq[m], inclusive='right')
                    new_time_coord = xr.concat([ds.time, xr.DataArray(additional_times, dims='time')], dim='time')
                    ds = ds.reindex(time=new_time_coord)
            
            if x is None: # [c, t, h, w]
                if 'time' in ds.dims:
                    x = np.empty([nr_bands, ds.sizes['time'], ds.sizes['y'], ds.sizes['x']])
                else:
                    x = np.empty([nr_bands, ds.sizes['y'], ds.sizes['x']])
            x[nr_band:nr_band+len(bands[m])] = ds.to_array().values

            if not master:
                master = ds
            ds.close()

            nr_band += len(bands[m])

        if len(x.shape) == 4:
            x = np.swapaxes(x, 0, 1) # reorder to [t, c, h, w]

        return torch.from_numpy(x).type(torch.float32), master

