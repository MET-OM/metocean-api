from __future__ import annotations
import os
from typing import TYPE_CHECKING
import xarray as xr
from tqdm import tqdm
from .aux_funcs import get_dates, get_url_info, get_near_coord, create_dataframe, get_tempfiles
if TYPE_CHECKING:
    from .ts_mod import TimeSeries  # Only imported for type checking

def echowave_ts(ts: TimeSeries, save_csv = False, save_nc = False, use_cache =False):
    """
    Extract times series of  the nearest gird point (lon,lat) from
    ECHOWAVE wave hindcast and save it as netcdf.
    source: https://data.4tu.nl/datasets/f359cd0f-d135-416c-9118-e79dccba57b9/1
    """
    #ts.variable.append('longitude') # keep info of regular lon
    #ts.variable.append('latitude')  # keep info of regular lat
    dates = get_dates(product=ts.product, start_date=ts.start_time, end_date=ts.end_time)

    tempfiles = get_tempfiles(ts.product,ts.lon, ts.lat, dates)

    selection = None
    lon_near = None
    lat_near = None

    # extract point and create temp files
    for i in tqdm(range(len(dates))):
        url = get_url_info(ts.product, dates[i])

        if i == 0:
            selection, lon_near, lat_near = get_near_coord(
                url, ts.lon, ts.lat, ts.product
            )
            tqdm.write(f'Nearest point to lat.={ts.lat:.3f},lon.={ts.lon:.3f} was found at lat.={lat_near:.3f},lon.={lon_near:.3f}')

        if use_cache and os.path.exists(tempfiles[i]):
            tqdm.write(f"Found cached file {tempfiles[i]}. Using this instead")
        else:
            with xr.open_dataset(url) as dataset:
                tqdm.write(f"Downloading {url}.")
                # Reduce to the wanted variables and coordinates
                dataset = dataset[ts.variable]
                dataset = dataset.sel(selection).squeeze(drop=True)
                dataset.to_netcdf(tempfiles[i], format='NETCDF3_64BIT') # add format to avoid *** AttributeError: NetCDF: String match to name in use

    __remove_if_datafile_exists(ts.datafile)

    # merge temp files
    with xr.open_mfdataset(tempfiles) as ds:
        if save_nc:
            # Save the unaltered structure
            ds = ds.sel({"time": slice(ts.start_time, ts.end_time)})
            ds.to_netcdf(ts.datafile.replace(".csv", ".nc"), format='NETCDF3_64BIT') # add format to avoid *** AttributeError: NetCDF: String match to name in use
        df = create_dataframe(
            product=ts.product,
            ds=ds,
            lon_near=lon_near,
            lat_near=lat_near,
            outfile=ts.datafile,
            variable=ts.variable[:2],
            start_time=ts.start_time,
            end_time=ts.end_time,
            save_csv=save_csv,
            height=ts.height,
        )

    if not use_cache:
        # remove temp/cache files
        __clean_cache(tempfiles)
    return df


def __clean_cache(tempfiles):
    for tmpfile in tempfiles:
        try:
            os.remove(tmpfile)
        except PermissionError:
            print(f"Skipping deletion of {tmpfile} due to PermissionError")

def __remove_if_datafile_exists(datafile):
    if os.path.exists(datafile):
        os.remove(datafile)
