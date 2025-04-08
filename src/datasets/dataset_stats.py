"""Compute statistics of a (big) dataset using the Dataset structure of torch, useful for normalization and data augmentation."""
import json
import torch
import numpy as np
import os
from tqdm import tqdm

def load_from_file(fn, data_dir, bands, return_tuple=2, overwrite=False):
    """load the statistics from a file."""
    # load each modalitiy from a separate file
    stats1 = []
    stats2 = []
    if bands:
        for m in bands:
            file_path = os.path.join(data_dir, m, fn)
            with open(file_path, 'r') as f:
                dic = json.load(f)
            if isinstance(dic, dict):
                for b in bands[m]:
                    if b and b in dic:
                        stats1.append(dic[b][0])
                        stats2.append(dic[b][1])
                    else:
                        raise ValueError(f'band {b} not in {file_path}')
            elif isinstance(dic, list): # assume it is list when band not in dic
                stats1.append(dic[0])
                stats2.append(dic[1])
            else:
                raise ValueError(f'unsupported format in {file_path}')
        # expand the stats to the same shape (assumed first one is the master)
        stats1 = [torch.tensor(s).expand_as(torch.tensor(stats1[0])) for s in stats1]
        stats2 = [torch.tensor(s).expand_as(torch.tensor(stats2[0])) for s in stats2]
        stats1 = torch.tensor(stats1)
        stats2 = torch.tensor(stats2)
    else:
        file_path = os.path.join(data_dir, fn)
        with open(file_path, 'r') as f:
            dic = json.load(f)
        stats1 = torch.tensor(dic[0])
        stats2 = torch.tensor(dic[1])
    res = (stats1, stats2)
    # overwrite part of the results
    if overwrite:
        for i in range(overwrite[0][0], overwrite[0][1]):
            if len(overwrite[1]) > 0:
                res[0][i] = overwrite[1][i]
            if len(overwrite[2]) > 0:
                res[1][i] = overwrite[2][i]

    # return a tuple when higher than len, otherwise use as index
    if return_tuple >= len(res) or return_tuple < 0:
        return res
    else:
        return res[return_tuple]

class DatasetStats():
    """Compute stats for a dataset, before apply transformations."""
    def __init__(self, dataset, target_name, band_dim, dims=[0,1,3,4]):
        # assumed shape for x (y without channel): (batch, time, channel, H, W)
        # hence default -> over everything but channel
        self.dm = dataset
        self.target_name = target_name # index of the data return by the dataloader
        if isinstance(dims, list):
            dims = tuple(dims)
        self.dims = dims
        self.band_dim = band_dim # dimension number of the bands

    def compute(self, type:str, fn=None, **kwargs):
        """compute the <type> statistics for the dataset and save them to a file if fn is given."""
        dl = self.dm.full_dataloader()
        if type == 'min_max':
            data = self.get_min_max(dl)
        elif type == 'mean_std':
            data = self.get_mean_std(dl)
        elif type == 'histogram':
            data = self.get_histogram(dl, **kwargs)
        else:
            raise ValueError(f'DatasetStats: type {type} not supported.')

        if fn:
            self.save_to_file(data, fn)

        return data

    def save_to_file(self, data, fn):
        """save the data to a file. If bands are given, save each modality in a separate file.
        data is a list of tensors to be saved."""
        idx = 0
        if self.dm.hparams.bands:
            for m in self.dm.hparams.bands:
                file_path = os.path.join(self.dm.hparams.data_dir, m, fn)
                m_bands = self.dm.hparams.bands[m]
                dic = {}
                for b in m_bands:
                    if len(data[0].shape) == 1 or self.band_dim is None:
                        # when there are no dimensions, assume it is a list of tensors
                        dic[b] = [data[i][idx].tolist() for i in range(len(data))]
                    else:
                        dic[b] = [data[i].select(dim=self.band_dim, index=idx).tolist() for i in range(len(data))]
                    idx += 1
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'w') as f:
                    json.dump(dic, f)
        else:
            file_path = os.path.join(self.dm.hparams.data_dir, fn)
            l = [d.tolist() for d in data]
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(l, f)

    def get_min_max(self, dl):
        """compute the min and max for the dataset. target_name is the name of the tuple the dl returns."""
        print(f'DatasetStats: {self.target_name} - calc min and max')
        d_min, d_max = np.inf, -np.inf
        rm_batch_dim = True

        for data in tqdm(dl):
            dims = self.dims
            d = data[self.target_name]
            # change data shape in case batch is not tensor (assumed can be concat along first dim)
            if isinstance(d, (tuple,list)) and self.dims and 0 in self.dims:
                d = torch.concat(d, dim=0)
                dims = tuple(np.array(self.dims[1:]) - 1)
                rm_batch_dim = False

            h_min = np.nanmin(d, axis=dims, keepdims=True)
            h_max = np.nanmax(d, axis=dims, keepdims=True)
            d_min = np.minimum(h_min, d_min)
            d_max = np.maximum(h_max, d_max)
        # return and remove batch dimension
        min, max = torch.tensor(d_min), torch.tensor(d_max)

        if rm_batch_dim:
            min, max = min.squeeze(dim=0), max.squeeze(dim=0)

        return min, max

    def get_mean_std(self, dl):
        """compute the mean and std for the dataset. target_name is the name of the tuple the dl returns."""
        print(f'DatasetStats: {self.target_name} - calc mean and std')
        channels_sum, channels_squared_sum, num_batches = 0, 0, 0
        rm_batch_dim = True
        for data in tqdm(dl):
            dims = self.dims
            d = data[self.target_name]
            # change data shape in case batch is not tensor (assumed can be concat along first dim)
            if isinstance(d, (tuple,list)) and self.dims and 0 in self.dims:
                d = torch.concat(d, dim=0)
                dims = tuple(np.array(self.dims[1:]) - 1)
                rm_batch_dim = False

            channels_sum += np.nanmean(d, axis=dims, keepdims=True)
            channels_squared_sum += np.nanmean(np.array(d)**2, axis=dims, keepdims=True)

            num_batches += 1

        mean = channels_sum / num_batches
        # std = sqrt(E[X^2] - (E[X])^2)
        std = torch.tensor((channels_squared_sum / num_batches - mean ** 2) ** 0.5)
        mean = torch.tensor(mean)
        if rm_batch_dim:
            mean, std = mean.squeeze(dim=0), std.squeeze(dim=0)

        return mean, std

    def get_histogram(self, dl, bins, min, max, normalize=False):
        print(f'DatasetStats: {self.target_name} - calc histogram')
        def calc_hist(d, dims, bins, min, max):
            new_dims = [h for h in range(len(d.shape)) if h not in dims]
            if len(new_dims) > 0:
                # reshape the tensor to [channel, -1]
                new_dims.extend(dims)
                d = d.permute(new_dims).flatten(start_dim=1)
                if not isinstance(min, (list, tuple)) and not isinstance(min, torch.Tensor):
                    min = [min] * d.shape[0]
                if not isinstance(max, (list, tuple)) and not isinstance(max, torch.Tensor):
                    max = [max] * d.shape[0]
                if not isinstance(bins, (list, tuple)):
                    bins = [bins] * d.shape[0]
                d_hist_list = []
                for j in range(d.shape[0]):
                    d_hist = torch.histc(d[j][~torch.isnan(d[j])], bins=bins[j], min=min[j], max=max[j])
                    d_hist_list.append(d_hist)
                d_hist = torch.stack(d_hist_list, axis=0)
            else:
                d = d.flatten()
                d_hist = torch.histc(d[~torch.isnan(d)], bins=bins, min=min, max=max)
                d_hist = torch.unsqueeze(d_hist,0) # add dimension for channel, so shape is same as with channels
            return d_hist
        
        if isinstance(min, (torch.Tensor)):
            min, max = min.squeeze(), max.squeeze()

        hist = None            
        for data in tqdm(dl):
            dims = self.dims
            d = data[self.target_name]
            # change data shape in case batch is not tensor (assumed can be concat along first dim)
            if isinstance(d, (tuple,list)) and self.dims and 0 in self.dims:
                d = torch.concat(d, dim=0)
                dims = tuple(np.array(self.dims[1:]) - 1)

            h_hist = calc_hist(d, dims, bins, min, max)
            hist = h_hist if hist is None else hist + h_hist

        if normalize:
            hist = hist / hist.sum()

        return [hist]
        
if __name__ == '__main__':
    import yaml
    import itertools
    import geopandas as gpd
    import rootutils
    home_folder = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    from src.datasets.datamodule import DataModule
    # load final geoids
    data_dir = os.path.join(home_folder, 'data/CropClimateX/')
    geoids = list(itertools.chain.from_iterable(yaml.safe_load(open(os.path.join(home_folder, 'configs/data/geoids_split.yaml')))['split']['ratio']))


    bands = [
        {'modis': [f'sur_refl_b0{i}' for i in range(1,5)] + ['sur_refl_b06']},
        {'landsat8': ['B02', 'B03', 'B04', 'B05', 'B06']},
        {'sen2': ["B02","B03","B04","B06","B08","B11"]},
        {'lai': ["Lai_500m", "Fpar_500m"]},
        {'lst': ['LST_Day_1km', 'LST_Night_1km']},
        {'daymet': ['tmax', 'tmin', 'prcp', 'vp', 'srad', 'hcw']},
        {'usdm': ['usdm']},
        {'cdl': ['cdl_30m']},
        {'cdl_500m': ['cdl_500m']},
        {'nlcd': ['nlcd']},
        ]
    dims=[0,1,3,4] # [batch, time, channel, H, W]
    band_dim = 1
    # uncomment this for stats of dem and soil data
    # bands = [
    #     {'dem': ['elevation', 'slope', 'aspect', 'curvature']},
    #     {'soil': ['bulk_density','cec','clay','ph','sand','silt','soc',]}
    # ]
    # dims=[0,2,3] # batch, channel, H, W
    # band_dim = 0 # without batch dimension
    for band in bands:
        print(band)
        if list(band.keys())[0] == 'sen2':
            geoid_pids = list(itertools.chain.from_iterable(yaml.safe_load(open(os.path.join(home_folder, 'configs/data/geoid_pids_split_sen2.yaml')))['split']['ratio']))
        else:
            geoid_pids = list(itertools.chain.from_iterable(yaml.safe_load(open(os.path.join(home_folder, 'configs/data/geoid_pids_split.yaml')))['split']['ratio']))

        dic = {'data_dir': data_dir,
            'task': 'extremes',
            'transformations': None,
            'bands': band,
            'geoids': geoid_pids,
            'num_workers': os.cpu_count()//8,
            'batch_size': 16,
            'handle_nans':False,
        }

        dm = DataModule(**dic)
        dm.setup()
        st = DatasetStats(dm, 'x', band_dim, dims=dims)
        st.compute('min_max', fn='min_max.json')
        st = DatasetStats(dm, 'x', band_dim, dims=dims)
        st.compute('mean_std', fn='mean_std.json')

    # compute yield stats
    for crop in ['corn','cotton','soybeans','winter_wheat','oats']:
        print(crop)
        dic = {'data_dir': data_dir,
            'transformations': None,
            'crops': [crop],
            'bands': {},
            'geoids': geoids,
        }
        dm = DataModule(**dic)
        dm.setup()
        st = DatasetStats(dm, 'y', None, dims=None)
        
        st.compute('min_max', fn=f'yield/{crop}_min_max.json')
        st.compute('mean_std', fn=f'yield/{crop}_mean_std.json')
