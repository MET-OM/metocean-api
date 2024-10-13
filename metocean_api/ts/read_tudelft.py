from __future__ import annotations
from pathlib import Path
import os
from typing import TYPE_CHECKING
import xarray as xr
import numpy as np
from .aux_funcs import get_date_list, get_url_info, get_near_coord, create_dataframe, check_datafile_exists
if TYPE_CHECKING:
    from .ts_mod import TimeSeries  # Only imported for type checking

def ECHOWAVE_ts(ts: TimeSeries, save_csv = False, save_nc = False, use_cache =False):
    """
    Extract times series of  the nearest gird point (lon,lat) from
    ECHOWAVE wave hindcast and save it as netcdf.
    source: https://data.4tu.nl/datasets/f359cd0f-d135-416c-9118-e79dccba57b9/1
    """
    #ts.variable.append('longitude') # keep info of regular lon
    #ts.variable.append('latitude')  # keep info of regular lat
    date_list = get_date_list(product=ts.product, start_date=ts.start_time, end_date=ts.end_time)        

    tempfiles = __tempfile_dir(ts.product,ts.lon, ts.lat, date_list,dirName="cache")


    # extract point and create temp files
    for i in range(len(date_list)):
        x_coor_str, y_coor_str, infile = get_url_info(product=ts.product, date=date_list[i])

        if i==0:
            x_coor, y_coor, lon_near, lat_near = get_near_coord(infile=infile, lon=ts.lon, lat=ts.lat, product=ts.product)

        if use_cache and os.path.exists(tempfiles[i]):
            print(f"Found cached file {tempfiles[i]}. Using this instead")
        else:
            with xr.open_dataset(infile) as dataset:
                # Reduce to the wanted variables and coordinates
                dataset = dataset[ts.variable]
                selection = {x_coor_str: x_coor, y_coor_str: y_coor}
                dataset = dataset.sel(selection).squeeze(drop=True)
                dataset.to_netcdf(tempfiles[i], format='NETCDF3_64BIT') # add format to avoid *** AttributeError: NetCDF: String match to name in use


    check_datafile_exists(ts.datafile)
    #merge temp files
    with xr.open_mfdataset(tempfiles) as ds:
        df = create_dataframe(product=ts.product,ds=ds, lon_near=lon_near, lat_near=lat_near, outfile=ts.datafile, variable=ts.variable[:2], start_time = ts.start_time, end_time = ts.end_time, save_csv=save_csv,save_nc = save_nc, height=ts.height)    

    if not use_cache:
        #remove temp/cache files
        __clean_cache(tempfiles)
    return df



def __clean_cache(tempfiles):
    for tmpfile in tempfiles:
        try:
            os.remove(tmpfile)
        except PermissionError:
            print(f"Skipping deletion of {tmpfile} due to PermissionError")

def __tempfile_dir(product,lon,lat, date_list,dirName):
    tempfile = [None] *len(date_list)
    # Create directory
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")
    
    for i in range(len(date_list)):
        tempfile[i] = str(Path(dirName+"/"+product+"_"+"lon"+str(lon)+"lat"+str(lat)+"_"+date_list.strftime('%Y%m%d')[i]+".nc"))

    return tempfile