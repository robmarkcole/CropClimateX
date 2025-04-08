# %%
import torch
import numpy as np
import glob
import os
import json
from joblib import Parallel, delayed
import zarr
import re
import warnings

class PrepDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, x_transform=None, y_transform=None, bands:dict=None,
                 pred_bands:dict=None, geoids=[], years=[], seed=None, return_meta_data=False,
                 handle_nans='none', **kwargs):
        super().__init__()
        if len(kwargs) > 0:
            warnings.warn(f'Unused kwargs: {kwargs}')
        self.data_dir = data_dir
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.bands = bands
        self.pred_bands = pred_bands
        self.x_c = None
        self.y_c = None
        self.seed = seed
        self.return_meta_data = return_meta_data
        self.handle_nans = handle_nans
        self.geoids = geoids
        
        if not geoids or len(geoids) == 0:
            raise ValueError('parameter geoids must be provided.')

        if not years or len(years) == 0:
            years = ['2018', '2019', '2020', '2021', '2022']
        self.years = years
       
        # filter with compiled regex
        geoid_patterns = [f"_{geoid}_" for geoid in geoids]
        year_patterns = [f"_{year}" for year in years]
        def process_chunk(files):
            """Process a chunk of files to find matches"""
            return [
                fn for fn in files
                if any(pat in fn for pat in geoid_patterns) 
                and any(pat in fn for pat in year_patterns)
            ]

        fns = glob.glob(os.path.join(data_dir, 'x*'))
        results = Parallel(n_jobs=os.cpu_count()*2)(delayed(process_chunk)(fns[i:i+1000]) for i in range(0, len(fns), 1000))        
        self.fns = [fn for chunk_result in results for fn in chunk_result]

        # load band mapping
        if self.bands:
            self.bands = [vv for v in bands.values() for vv in v]
            with open(data_dir + 'band_name_mapping.json') as fh:
                bands = json.load(fh)
            if not all(value in bands for value in self.bands):
                raise ValueError(f'not all bands are provided by the dataset: {self.bands}')
            self.band_idx = [bands[k] for k in self.bands]
            if 'channel_dim' in bands:
                # get channel dim
                self.x_c = int(bands['channel_dim'])
        if self.pred_bands:
            self.pred_bands = [vv for v in pred_bands.values() for vv in v]
            with open(data_dir + 'pred_band_name_mapping.json') as fh:
                pred_bands = json.load(fh)
            if not all(value in pred_bands for value in self.pred_bands):
                raise ValueError(f'not all pred_bands are provided by the dataset: {self.pred_bands}')
            self.pred_band_idx = [pred_bands[k] for k in self.pred_bands]
            if 'channel_dim' in pred_bands:
                # get channel dim
                self.y_c = int(pred_bands['channel_dim'])

    def __len__(self):
        return len(self.fns)
    
    def __getitem__(self, item):
        fn_mask = self.fns[item].replace('x_', 'mask_')
        fn_y = self.fns[item].replace('x_', 'y_')

        x = self.load_file(self.fns[item])
        # select bands
        if self.bands and not self.x_c is None:
            x = np.take(x, self.band_idx, axis=self.x_c)
        # handle nans
        if self.handle_nans == 'zeros':
            x = np.where(np.isnan(x), 0, x)
        
        x = torch.from_numpy(x).type(torch.float32)
        if self.x_transform:
            x = self.x_transform(x)
        
        out = {'x':x}
        
        if os.path.exists(fn_y):
            y = self.load_file(fn_y)
            if self.pred_bands and not self.y_c is None:
                y = np.take(y, self.pred_band_idx, axis=self.y_c)
            if self.handle_nans == 'zeros':
                y = np.where(np.isnan(y), 0, y)
            y = torch.from_numpy(y).type(torch.float32)
            if self.y_transform:
                y = self.y_transform(y)
            out['y'] = y

        if os.path.exists(fn_mask):
            mask = self.load_file(fn_mask)
            mask = torch.from_numpy(mask).type(torch.float32)
            out['mask'] = mask

        if self.return_meta_data:
            # add meta data
            file_name = os.path.basename(self.fns[item]).split('.')[0].split('_')
            if len(file_name) == 5:
                out['meta'] = {'time':file_name[-1], 'geoid':file_name[1], 'pid':file_name[2], 'split_part':file_name[3]}
            else:
                out['meta'] = {'time':file_name[-1], 'geoid':file_name[1]}

        return out

    def load_file(self, fn):
        if fn.split('.')[-1] == 'npy':
            return np.load(fn)
        elif fn.split('.')[-1] == 'zarr':
            z = zarr.open(fn, mode='r')
            if isinstance(z, zarr.Array):
                return np.array(z)
            else:
                arr = np.array(z['arr_0'].astype(np.float32))
                if 'fill_value' in z:
                    fill_value = z['fill_value'][()]
                    arr[arr == fill_value] = np.nan
                return arr
        else:
            raise ValueError('file format not supported.')

