import glob
import os
import warnings

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon

import rioxarray as rxr
import xarray as xr
import rasterio as rio
from rasterio import features
from rasterstats import zonal_stats
import itertools

from utils import load_subregion_and_reproject, rm_attrs


def load_cdl_patch(shp, path='data/cropland/', classes=None, confidence=.6, time_slice=('2008-01-01', '2022-12-31'), crs=None):
    start_year, end_year = int(time_slice[0].split('-')[0]), int(time_slice[1].split('-')[0])
    ds_l = []
    for year in range(start_year, end_year+1):
        fn = os.path.join(path, 'usda_cdl_national', f'{year}', f'{year}_30m_cdls.tif')
        fn_conf = os.path.join(path, 'confidence', f'{year}', f'{year}_30m_confidence_layer.img')

        if os.path.exists(fn):
            ds = rxr.open_rasterio(fn)
            if os.path.exists(fn_conf):
                ds_conf = rxr.open_rasterio(fn_conf)
            elif os.path.exists(fn_conf.replace('.img', '.tif')):
                ds_conf = rxr.open_rasterio(fn_conf.replace('.img', '.tif'))
            else:
                continue
            ds = load_subregion_and_reproject(ds, shp, ds.rio.crs, crs, 30, all_touched=False)
            ds_conf = load_subregion_and_reproject(ds_conf, shp, ds_conf.rio.crs, crs, 30, all_touched=False)
            ds = xr.concat([ds, ds_conf], dim='band')
            ds = ds.to_dataset(name='cdl_30m')
            # add the time coord to concat
            ds = ds.assign_coords(time=pd.to_datetime([f'{year}-01-01']))
            ds_l.append(ds)

    if len(ds_l) > 1:
        ds = xr.concat(ds_l, dim='time')
    else:
        ds = ds_l[0]
    # remove pixels with confidence below the threshold
    if confidence and isinstance(confidence, (int, float)):
        if not 0 <= confidence <= 1:
            raise ValueError('confidence should be a float between 0 and 1')
        ds = ds.where(ds.isel(band=1) >= confidence)
    ds = ds.isel(band=0) # only use crop layer
    ds = ds.drop_vars('band')
    if classes: # remove classes not in the list
        ds = ds.where(ds.isin(list(classes.values())))

    ds['cdl_30m'] = ds['cdl_30m'].astype('int8') # save memory by using int8

    # rm and set attributes
    ds = rm_attrs(ds)
    if classes:
        ds.attrs['cropland_class_names'] = list(classes.keys())
        ds.attrs['cropland_class_values'] = list(classes.values())
    return ds

def create_grid_from_coords(ds):
    
    x_resolution = abs(ds.rio.resolution()[0])
    y_resolution = abs(ds.rio.resolution()[1])
    
    x_min, y_min, x_max, y_max = ds.rio.bounds()
    xx = np.arange(x_min, x_max, x_resolution)
    yy = np.arange(y_min, y_max, y_resolution)
    polys = []
    for x, y in itertools.product(xx, yy):
        polygon = Polygon([(x, y), (x+x_resolution, y), (x+x_resolution, y+y_resolution), (x, y+y_resolution)])
        polys.append(polygon)

    fishnet = gpd.GeoDataFrame(geometry=polys, columns=['geometry']).set_crs(ds.rio.crs)
    return fishnet

def create_coarse_cdl_map(ref_ds, cropland, threshold=60, num_pixels=289):
    """
    description
        downsamples a raster to a coaser resolution
        using a reference raster
        
    args
        ref_ds(xarray Dataset): reference raster
        cropland(xarray DataArray): high resolution raster to be downsampled
        threshold(float): percentage coverage of cropland in each grid of ref_ds
        num_pixels(int): number of high resolution raster pixels in each grid of 
                         reference raster e.g. 289 for 30m pixels in 500m grid
    """
    
    # check for equal crs
    assert ref_ds.rio.crs == cropland.rio.crs, "CRS are not equal"
    
    fishnet = create_grid_from_coords(ref_ds)
    # --create column per raster category, and pixel count as value
    zs = zonal_stats(vectors=fishnet['geometry'], raster=cropland.values.squeeze(), affine=cropland.rio.transform(), categorical=True, nodata=np.iinfo(np.int8).min)
    stats = pd.DataFrame(zs).fillna(0)
    results = pd.merge(left=fishnet, right=stats, how='left', left_index=True, right_index=True)

    # --find class with max pixels exceeding threshold
    results['class'] = results[results.columns.difference(["geometry"])].idxmax(axis=1)
    results['class_cover'] = results[results.columns.difference(["geometry", 'class'])].max(axis=1)
    results['class'] = np.where(results['class_cover'] >= threshold*num_pixels/100.0, results['class'], 0) 

    # rasterize
    # create a generator of geom, value pairs to use in rasterizing
    shapes = ((geom,value) for geom, value in zip(results.geometry, results['class']))
    results_rasterized = features.rasterize(shapes=shapes, fill=0, out=np.zeros(tuple(ref_ds.sizes[d] for d in ['y', 'x'])), transform=ref_ds.rio.transform())

    # convert to dataarray
    results_rasterized = xr.DataArray(results_rasterized, coords={'y':ref_ds.coords['y'], 'x':ref_ds.coords['x']}, attrs=cropland.attrs)
    results_rasterized.rio.write_crs(ref_ds.rio.crs, inplace=True)  
    
    return results_rasterized

def load_nlcd_patch(shp, path='data/landcover/usa_nlcd/', classes=None, time_slice=('2008-01-01', '2022-12-31'), crs=None):
    start_year, end_year = int(time_slice[0].split('-')[0]), int(time_slice[1].split('-')[0])
    ds_l = []
    for year in range(start_year, end_year+1):
        fns = glob.glob(os.path.join(path, f'nlcd_{year}_land_cover*.img'))
        if len(fns) > 0:
            if len(fns) > 1:
                warnings.warn(f'More than one nlcd file found for year {year}. Only the first file will be used.')

            fn = fns[0]
            ds = rxr.open_rasterio(fn)
            ds = load_subregion_and_reproject(ds, shp, ds.rio.crs, crs, 30, all_touched=False)
            ds = ds.to_dataset(name='nlcd')
            ds = ds.assign_coords({'time':pd.to_datetime([f'{year}-01-01'])})
            ds_l.append(ds)

    if len(ds_l) > 1:
        ds = xr.concat(ds_l, dim='time')
    else:
        ds = ds_l[0]
    ds = ds.drop_vars('band').squeeze('band')
    if classes: # remove classes not in the list
        ds = ds.where(ds.isin(list(classes.values())), other=0)
    ds = rm_attrs(ds)
    if classes:
        ds.attrs['landcover_class_names'] = list(classes.keys())
        ds.attrs['landcover_class_values'] = list(classes.values())

    return ds
