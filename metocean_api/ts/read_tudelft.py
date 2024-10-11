from abc import ABC, abstractmethod

import xarray as xr
import pandas as pd
import numpy as np
import time
import os
from pathlib import Path

from .aux_funcs import *

def ECHOWAVE_ts(self, save_csv = False, save_nc = False, save_cache =False):
    """
    Extract times series of  the nearest gird point (lon,lat) from
    ECHOWAVE wave hindcast and save it as netcdf.
    source: https://data.4tu.nl/datasets/f359cd0f-d135-416c-9118-e79dccba57b9/1
    """
    self.variable.append('longitude') # keep info of regular lon
    self.variable.append('latitude')  # keep info of regular lat
    date_list = get_date_list(product=self.product, start_date=self.start_time, end_date=self.end_time)        

    tempfile = tempfile_dir(self.product,self.lon, self.lat, date_list,dirName="cache")

    # extract point and create temp files
    for i in range(len(date_list)):
        x_coor_str, y_coor_str, infile = get_url_info(product=self.product, date=date_list[i])
             
        if i==0:
            x_coor, y_coor, lon_near, lat_near = get_near_coord(infile=infile, lon=self.lon, lat=self.lat, product=self.product)        

        ds = xr.open_dataset(infile)
        # Only download the variables we want
        ds = ds[self.variable]
        selection = {'longitude': lon_near, 'latitude': lat_near}
        ds = ds.sel(selection)
        ds.to_netcdf(tempfile[i],format='NETCDF3_64BIT') # add format to avoid *** AttributeError: NetCDF: String match to name in use


    check_datafile_exists(self.datafile)
    #merge temp files
    ds = xr.open_mfdataset(paths=tempfile[:])   
    ds.to_netcdf(f'{self.product}_xarray.nc',format='NETCDF3_64BIT') # add format to avoid *** AttributeError: NetCDF: String match to name in use
    
    #Save in csv format    
    df = create_dataframe(product=self.product,ds=ds, lon_near=lon_near, lat_near=lat_near, outfile=self.datafile, variable=self.variable[:2], start_time = self.start_time, end_time = self.end_time, save_csv=save_csv, save_nc = save_nc, height=self.height)    
    ds.close()
   #remove temp/cache files
    if save_cache == False:
        for i in range(len(date_list)):
            try:
                os.remove(tempfile[i])
            except PermissionError:
                print(f"Skipping deletion of {tempfile[i]} due to PermissionError")

    return df



def tempfile_dir(product,lon,lat, date_list,dirName):
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