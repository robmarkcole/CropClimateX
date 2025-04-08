import datetime
import xarray as xr

def harmonize_pc_sentinel2_dataset_to_old(data):
    cutoff = datetime.datetime(2022, 1, 25) # after 25.01.2022 the data has an offset
    offset = 1000
    bands = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B10","B11","B12",]

    # select the band which have an offset
    to_process = list(set(bands) & set(data.data_vars.keys()))
    for band in to_process:
        da_slice = data.sel(time=slice(cutoff, None))
        if len(da_slice.time) == 0:
            break # no data after the cutoff, assumed that other data vars have same time steps
        # Replace values for the selected time slice
        relevant_times = da_slice.time.values
        data[band].loc[{'time': data.time.isin(relevant_times)}] = data[band].loc[{'time': data.time.isin(relevant_times)}].clip(offset) - offset
    return data

def align_processing_version(ds, geoid_pid):
    if any([(int(str(id).split('_')[3][1:]) < 300) or (9999 == int(str(id).split('_')[3][1:])) for id in ds.id.values]):
        for t in range(len(ds.time)):
            da = ds.isel(time=t)
            version = int(str(da.id.values).split('_')[3][1:])
            if version < 300 or 9999 == version:
                # is 2.x adjust to 05.xx
                da = harmonize_s2_to_new(da)
                ds[{'time': t}] = da
                with open('harmonized.txt', 'a') as f:
                    f.write(f"{geoid_pid}: {da.time.dt.strftime('%Y-%m-%d').values} {id} to 05.xx\n")
    return ds

def harmonize_s2_to_new(da):
    bands = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B10","B11","B12",]
    offset = 1000
    to_process = list(set(bands) & set(da.data_vars.keys()))
    for band in to_process:
        # add offset and clip to make sure there is no overflow
        mask = da[band] != 0
        da[band] = da[band].clip(0, 65535 - offset) + offset
        da[band] = da[band].where(mask, 0)
    return da