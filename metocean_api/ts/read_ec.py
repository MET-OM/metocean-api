from abc import ABC, abstractmethod

import pandas as pd
import xarray as xr
import os 

from .aux_funcs import get_date_list, create_dataframe

def ERA5_ts(self, save_csv = False):
    """
    Extract times series of  the nearest gird point (lon,lat) from
    ERA5 reanalysis and save it as netcdf.
    """
    filename = download_era5_from_cds(self.start_time, self.end_time, self.lon, self.lat,self.variable, folder='temp')
    ds = xr.open_dataset(filename)
    df = create_dataframe(product=self.product,ds=ds, lon_near=ds.longitude.values[0], lat_near=ds.latitude.values[0], outfile=self.datafile, variable=self.variable,save_csv=save_csv, height=self.height)    
    
    return df

def download_era5_from_cds(start_time, end_time, lon, lat, variable,  folder='temp') -> str:
    import cdsapi
    """Downloads ERA5 data from the Copernicus Climate Data Store for a
    given point and time period"""
    start_time = pd.Timestamp(start_time)
    end_time = pd.Timestamp(end_time)
    c = cdsapi.Client()


    # Create directory
    try:
        # Create target Directory
        os.mkdir(folder)
        print("Directory " , folder ,  " Created ")
    except FileExistsError:
        print("Directory " , folder ,  " already exists")

    filename = f'{folder}/EC_ERA5.nc'

    days = get_date_list('ERA5',start_time, end_time)
    # Create string for dates
    dates = [days[0].strftime('%Y-%m-%d'), days[-1].strftime('%Y-%m-%d')]
    dates = '/'.join(dates)

    cds_command = {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': variable,
        'date': dates,
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
        'area': [
            lat+0.001, lon-0.001, lat-0.001,lon+0.001,
            #53.33, 1.31, 53.31,1.33,
        ],
    }

    c.retrieve('reanalysis-era5-single-levels', cds_command, filename)
    return filename

