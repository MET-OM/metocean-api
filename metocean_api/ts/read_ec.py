from abc import ABC, abstractmethod

import cdsapi
import pandas as pd

from .aux_funcs import get_date_list


def download_era5_from_cds(start_time, end_time, lon, lat, folder='temp') -> str:
    """Downloads ERA5 data from the Copernicus Climate Data Store for a
    given point and time period"""
    start_time = pd.Timestamp(start_time)
    end_time = pd.Timestamp(end_time)
    c = cdsapi.Client()

    filename = f'{folder}/EC_ERA5.nc'

    days = get_date_list('ERA5',start_time, end_time)
    # Create string for dates
    dates = [days[0].strftime('%Y-%m-%d'), days[-1].strftime('%Y-%m-%d')]
    dates = '/'.join(dates)

    cds_command = {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            '100m_u_component_of_wind', '100m_v_component_of_wind', '10m_u_component_of_wind',
            '10m_v_component_of_wind', '2m_temperature', 'instantaneous_10m_wind_gust',
            'mean_direction_of_total_swell', 'mean_direction_of_wind_waves', 'mean_period_of_total_swell',
            'mean_period_of_wind_waves', 'mean_wave_direction', 'mean_wave_period',
            'peak_wave_period', 'significant_height_of_combined_wind_waves_and_swell', 'significant_height_of_total_swell',
            'significant_height_of_wind_waves',
        ],
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
            lat+0.01, lon[0]-0.01, lat[0]-0.01,lon[1]+0.01,
            #53.33, 1.31, 53.31,1.33,
        ],
    }

    c.retrieve('reanalysis-era5-single-levels', cds_command, filename)
    return filename

