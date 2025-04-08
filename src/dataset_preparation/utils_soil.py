import pandas as pd
import xarray as xr
import rioxarray as rxr
import numpy as np
import os
from tqdm import tqdm
from utils import crop_to_region, load_subregion_and_reproject, rm_attrs

def prepare_soil_table(csv_path, column_names):
    """Params csv_path(str): file path of gnatsgo chorizon table.

    returns pandas table of soil properties averaged depthwise until 200cm
    """
    # read table
    soil_property = pd.read_csv(csv_path, index_col=False, usecols = column_names)

    # order columns
    soil_property = soil_property[column_names]

    # filter by horizon depth 200cm onwards
    if 'hzdepb_r' in column_names:
        soil_property = soil_property.loc[soil_property['hzdepb_r'] <= 200]
        soil_property.drop(['hzdepb_r'], axis=1, inplace=True)

    #  convert negative log values to original concentrations
    if 'ph1to1h2o_r' in column_names:
        soil_property['ph1to1h2o_r'] = 10**(-soil_property['ph1to1h2o_r'])

    if 'cokey' in column_names:
        # remove nan in cokey
        soil_property.dropna(subset=['cokey'], inplace=True)

        # convert cokey to numeric
        soil_property["cokey"] = soil_property["cokey"].astype(int)

        # average horizon/depth values per component
        soil_property = soil_property.groupby(['cokey'], as_index=False).mean()

    return soil_property


def create_soil_raster(soil_raster, chorizon_table, component_table):
    """Params soil_raster(xarray object): gnastgo raster chorizon_table(df object): chorizon table
    component_table(df object): component_table table.

    returns xarray object containing soil properties
    """
    # convert to numeric
    chorizon_table["cokey"] = chorizon_table["cokey"].astype(int)
    component_table["cokey"] = component_table["cokey"].astype(int)
    component_table["mukey"] = component_table["mukey"].astype(int)

    # merge chorizon and component
    merged_table = pd.merge(chorizon_table, component_table, on='cokey', how='left')

    # get list of chorizon columns without key
    chorizon_columns = list(chorizon_table.columns)
    chorizon_columns.remove('cokey')

    # calculate weighted average across components
    for col in chorizon_columns:
        merged_table[col] = merged_table[col] * merged_table['comppct_r'] / 100.0 

    # drop cokey and comppct_r
    merged_table.drop(['cokey', 'comppct_r'], axis=1, inplace=True)

    # find mean by mukey
    merged_table = merged_table.groupby(['mukey'], as_index=False).sum()

    # convert concentration back to neg log
    if 'ph1to1h2o_r' in merged_table:
        merged_table['ph1to1h2o_r'] = -np.log10(merged_table['ph1to1h2o_r'])

    # join table to soil raster by mukey
    soil_raster['mukey'] = soil_raster['mukey'].fillna(-1)
    mukey_raster_df = soil_raster['mukey'].to_dataframe(name='mukey').reset_index()
    mukey_raster_df = mukey_raster_df.astype({"mukey": int})

    # merge dfs on the 'keys' column
    merged_mukey_df = pd.merge(mukey_raster_df, merged_table, on='mukey', how='left')
    merged_mukey_df.drop(columns=['mukey'], inplace=True)

    # extract x and y coordinates
    x_coords = merged_mukey_df['x'].unique()
    y_coords = merged_mukey_df['y'].unique()
    for col in chorizon_columns:
        soil_raster[col] = xr.DataArray(merged_mukey_df[col].values.reshape(len(y_coords), len(x_coords), 1), dims=('y', 'x', 'time'), coords = {item: soil_raster.coords[item] for item in soil_raster.coords})

    return soil_raster

def load_soil_grid_patch(region, fn):
    da = rxr.open_rasterio(fn)
    da = load_subregion_and_reproject(da, region, da.rio.crs, region.crs, 250, all_touched=False)
    ds = da.to_dataset(dim="band")
    ds = ds.rename({old: new for old, new in zip(ds.data_vars, da.attrs['long_name'])})
    if 'FILL_MASK' in ds.data_vars:
        ds = ds.drop_vars('FILL_MASK')
    # replace -inf with nan
    ds = ds.where(ds != -np.inf)
    ds = rm_attrs(ds)
    return ds

def load_USGS_3DEP_patch(region, fn):
    import xrspatial
    from utils import rm_attrs
    da = rxr.open_rasterio(fn)
    da = load_subregion_and_reproject(da, region, da.rio.crs, region.crs, 30, all_touched=False)
    ds = da.to_dataset(dim="band")
    ds = ds.rename({1:'elevation'})
    ds = rm_attrs(ds)
    # slope, aspect, curvature
    ds['slope'] = xrspatial.slope(ds['elevation'])
    ds['aspect'] = xrspatial.aspect(ds['elevation'])
    ds['curvature'] = xrspatial.curvature(ds['elevation'])
    return ds

def download_USGS_3DEP(download_dir = 'data/usgs_dem'):
    import boto3
    from botocore import UNSIGNED
    from botocore.client import Config

    # the USGS_Seamless_DEM_1.vrt needs to be downloaded seperatly
    # and the paths in it need to be adjusted to the local paths

    # Initialize the S3 client with anonymous access
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    # Bucket name and prefix (subfolder)
    bucket_name = 'prd-tnm'
    prefix = 'StagedProducts/Elevation/1/TIFF/current/'

    # Create a local directory to store the downloaded files
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    # List and download TIFF files
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for content in tqdm(page.get('Contents', [])):
            key = content['Key']
            if key.endswith('.tif') or key.endswith('.tiff'):
                file_name = os.path.join(download_dir, os.path.basename(key))
                if os.path.exists(file_name):
                    print(f"Skipping {key} as {file_name} already exists")
                    continue
                print(f"Downloading {key} to {file_name}")
                s3.download_file(bucket_name, key, file_name)

    print("Download complete!")