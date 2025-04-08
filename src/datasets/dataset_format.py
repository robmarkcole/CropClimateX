import json
import torch
import os
import glob
import hydra
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf,open_dict
import rootutils
home_folder = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.datasets.datamodule import DataModule

def save(fn, data, dtype, format='zarr'):
    if not 'float' in dtype:
        # use highest number for nan
        fill_value = np.iinfo(dtype).max
        data = np.nan_to_num(data, nan=fill_value)
    else:
        fill_value = np.nan

    data = data.astype(dtype)

    if format == 'npy':
        np.save(fn + '.npy', data)
    elif format == 'npz':
        np.savez_compressed(fn + '.npz', data)
    elif format == 'zarr':
        import zarr
        zarr.save(fn + '.zarr', data, dtype=dtype, fill_value=fill_value)

def format_batch(batch, fo, dtype):
    """save a batch into separate files."""           
    for i in range(batch['x'].shape[0]):
        meta = batch['label']
        pid = f"_{meta['pid'][i]}" if 'pid' in meta else ''
        split_part = f"_{meta['split_part'][i]}" if 'split_part' in meta else ''
        fn = f"{meta['geoid'][i]}{pid}{split_part}_{meta['time'][i]}"

        x = batch['x'][i].numpy()
        save(os.path.join(fo, "x_" + fn),x,dtype)
        if 'mask' in batch:
            mask = batch['mask'][i].numpy()
            save(os.path.join(fo, "mask_" + fn),mask, 'uint8')
        if 'y' in batch and batch['y'].shape[0] > i:
            y = batch['y'][i].numpy()
            save(os.path.join(fo, "y_" + fn),y,dtype)

def format_list(batches, fo, dtype):
    """save each sample in a separate file with the meta data as name."""
    for i in range(len(batches['x'])):
        x = batches['x'][i].numpy()
        y = batches['y'][i].numpy()
        fn = f"{batches['label'][i]['geoid']}_{batches['label'][i]['time']}"
        if 'mask' in batches:
            mask = batches['mask'][i].numpy()
            save(os.path.join(fo, "mask_" + fn), mask, 'uint8')
        save(os.path.join(fo, "x_" + fn), x,dtype)
        save(os.path.join(fo, "y_" + fn), y,dtype)

def dataset_format_to_np(dm, fo, bands=None, pred_bands=None, bands_channel_dim=None, pred_bands_channel_dim=None, dtype='int8'):
    """preprocess the whole dataset to npy files. Each sample is saved in a separate file with the meta data as name."""
    # write band data to json
    if bands is not None:
        fn_dict = os.path.join(fo, 'band_name_mapping.json')
        band_dict = {k:i for i,k in enumerate(bands)}
        if bands_channel_dim:
            band_dict['channel_dim'] = bands_channel_dim
        with open(fn_dict, "w") as file:
            json.dump(band_dict, file)

    if pred_bands is not None:
        fn_dict = os.path.join(fo, 'pred_band_name_mapping.json')
        band_dict = {k:i for i,k in enumerate(pred_bands)}
        if pred_bands_channel_dim:
            band_dict['channel_dim'] = pred_bands_channel_dim
        with open(fn_dict, "w") as file:
            json.dump(band_dict, file)

    dl = dm.full_dataloader()
    print('nr samples:',len(dl.dataset.labels))
    dl.dataset.labels.to_csv(os.path.join(fo, 'labels.csv'), index=False)

    # load already processed labels
    files = [entry.name for entry in os.scandir(fo) if entry.name.startswith('x_')]
    files = [f.split('.')[0][2:] for f in files]
    col = 'geoid' if 'geoid' in dl.dataset.labels.columns else 'GEOID_PID'
    def lam_func(x):
        out = f"{x[col]}_"
        if 'split_part' in x:
            out += f"{x['split_part']}_"
        if 'time_range' in x:
            out += f"{x['time_range'][0]}-{x['time_range'][-1]}"
        elif 'year' in x:
            out += f"{x['year']}"
        return out

    dl.dataset.labels['fn'] = dl.dataset.labels.apply(lambda x: lam_func(x), axis=1)
    # rm already processed labels
    dl.dataset.labels = dl.dataset.labels[~dl.dataset.labels['fn'].isin(files)]
    print('nr samples left:',len(dl.dataset.labels))

    # load the dataset
    for batch in tqdm(dl, desc="Processing dataset", total=len(dl)):
        # save each sample in a separate file with the meta data as name
        if isinstance(batch['x'], torch.Tensor): # spatial dataset
            format_batch(batch, fo, dtype)
        elif isinstance(batch['x'], (list,tuple)): # yield dataset
            format_list(batch, fo, dtype)

def main_wrapper(ds_name, **kwargs):
    @hydra.main(version_base="1.3", config_path="../../configs", config_name="main.yaml")
    def main(cfg):
        data_dir = cfg.data['data_dir']
        fo = os.path.join(data_dir, 'prep_datasets', ds_name)
        if os.path.exists(fo):
            i = input(f"Folder {fo} already exists, will remove already existing ids from DataLoader. Press any key to continue.")
        os.makedirs(fo, exist_ok=True)
        dm = instantiate_cfg(cfg)
        OmegaConf.save(cfg, os.path.join(fo,"config.yaml"))
        start_time = pd.Timestamp.now()
        dataset_format_to_np(dm=dm, fo=fo, **kwargs)
        print(f"Finished in {(pd.Timestamp.now()-start_time).seconds/60} minutes.")
    main()

def instantiate_cfg(cfg):
    with open_dict(cfg):
       cfg.data.return_label_data = True
    LightningDataModule = hydra.utils.instantiate(cfg.data, _convert_="object")
    LightningDataModule.setup()
    return LightningDataModule

# use like this: python dataset_format.py ds_name=<new-folder-name/dataset-name> data=<dataset-yaml-filename> bands=<band-names> pred_bands=<pred-band-names>
#                bands_channel_dim=<channel-dim> pred_bands_channel_dim=<pred-channel-dim> dtype=<data-type>
# example:
# python src/datasets/dataset_format.py --ds_name=pre_drought_landsat_uint16 --data=drought_landsat_data_prep --bands B02 B03 B04 B05 B06 elevation slope aspect curvature --pred_bands usdm_mode --bands_channel_dim=-3 --dtype=uint16

# python src/datasets/dataset_format.py --ds_name=pre_yield_modis_corn_float32 --data=yield_modis_data_prep --bands sur_refl_b01 sur_refl_b02 sur_refl_b03 sur_refl_b04 sur_refl_b06 tmax tmin prcp vp srad elevation slope aspect curvature bulk_density cec clay ph sand silt soc --pred_bands yield --bands_channel_dim=-3 --dtype=float32

if __name__ == '__main__':
    os.environ['HYDRA_FULL_ERROR'] = '1'
    p = argparse.ArgumentParser()
    p.add_argument("--ds_name", type=str, default='NPYDataset', help="Name of the dataset")
    p.add_argument("--data", type=str, default=None, help="Name of the dataset yaml file")
    p.add_argument("--bands", nargs='+', help="Band names to be saved in the dataset")
    p.add_argument("--pred_bands", nargs='+', help="Prediction band names to be saved in the dataset")
    p.add_argument("--bands_channel_dim", type=str, help="Channel dimension of bands")
    p.add_argument("--pred_bands_channel_dim", type=str, help="Channel dimension of pred_bands")
    p.add_argument("--dtype", type=str, default='int8', help="Type of the data to be saved")
    args = p.parse_args()
    sys.argv = [sys.argv[0]] # rm all arguments for hydra
    sys.argv.append('data=' + args.__dict__.pop('data')) # add data to sys.argv for hydra
    main_wrapper(**args.__dict__)
