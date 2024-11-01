from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, List
import os
import subprocess
import pandas as pd
import xarray as xr
import numpy as np
from ..product import Product

from .. import aux_funcs

if TYPE_CHECKING:
    from ts.ts_mod import TimeSeries  # Only imported for type checking


def find_product(name: str) -> Product:
    match name:
        case "ERA5":
            return ERA5(name)
        case "GTSM":
            return GTSM(name)
    return None


class ERA5(Product):

    def import_data(self, ts: TimeSeries, save_csv=True, save_nc=False, use_cache=False):
        """
        Extract times series of  the nearest gird point (lon,lat) from
        ERA5 reanalysis and save it as netcdf.
        """
        if ts.variable == []:
            ts.variable = [
                "100m_u_component_of_wind",
                "100m_v_component_of_wind",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_temperature",
                "instantaneous_10m_wind_gust",
                "mean_direction_of_total_swell",
                "mean_direction_of_wind_waves",
                "mean_period_of_total_swell",
                "mean_period_of_wind_waves",
                "mean_wave_direction",
                "mean_wave_period",
                "peak_wave_period",
                "significant_height_of_combined_wind_waves_and_swell",
                "significant_height_of_total_swell",
                "significant_height_of_wind_waves",
            ]
        filenames = self.__download_era5_from_cds(ts.start_time, ts.end_time, ts.lon, ts.lat,ts.variable, folder='cache')
        df_res = None
        ds_res = None
        variable_info = []

        for filename in filenames:
            with xr.open_mfdataset(filename) as ds:
                # Rename the dimensions and variables to match the required format
                ds = ds.rename_dims({"valid_time": "time"}).rename_vars({"valid_time": "time"})
                lon_near=ds.longitude.values[0]
                lat_near=ds.latitude.values[0]
                ds = ds.drop_vars(['longitude','latitude'], errors="ignore")
                df = aux_funcs.create_dataframe(self.name, ds, lon_near, lat_near, ts.datafile, ts.start_time, ts.end_time, save_csv=False)
                df.drop(columns=['number', 'expver'], inplace=True, errors='ignore')
                variable = df.columns[0]
                try:
                    standard_name = ds[variable].standard_name
                except AttributeError:
                    standard_name = '-'
                try:
                    long_name = ds[variable].long_name
                except AttributeError:
                    long_name = '-'
                variable_info.append(f'#{variable};{standard_name};{long_name};{ds[variable].units}\n')

                if df_res is None:
                    df_res = df
                    ds_res = ds
                else:
                    df_res = df_res.join(df)
                    ds_res = ds_res.merge(ds, compat='override')

        if save_csv:
            lon_near = ds.longitude.values[0]
            lat_near = ds.latitude.values[0]
            top_header = f'#{ts.product};LONGITUDE:{lon_near:0.4f};LATITUDE:{lat_near:0.4f}\n'
            header = [top_header, '#Variable_name;standard_name;long_name;units\n'] + variable_info
            with open(ts.datafile, "w", encoding="utf8", newline="") as f:
                f.writelines(header)
                df_res.to_csv(f, header=False, encoding=f.encoding, index_label="time")

        if save_nc:
            ds_res.to_netcdf(ts.datafile.replace('csv','nc'))

        return df_res

    def __download_era5_from_cds(self,start_time, end_time, lon, lat, variable,  folder='cache', use_cache=True) -> str:
        """
        Downloads ERA5 data from the Copernicus Climate Data Store for a
        given point and time period
        """
        import cdsapi
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

        days =  pd.date_range(start=start_time , end=end_time, freq='D')
        # Create string for dates
        dates = [days[0].strftime('%Y-%m-%d'), days[-1].strftime('%Y-%m-%d')]
        dates = '/'.join(dates)
        filename_list = []
        for i in range(len(variable)):
            filename = f'{folder}/ERA5_'+"lon"+str(lon)+"lat"+str(lat)+"_"+days[0].strftime('%Y%m%d')+'_'+days[-1].strftime('%Y%m%d')+'_'+variable[i]+".nc"
            filename_list = np.append(filename_list,filename)
            cds_command = {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': variable[i],
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
            if use_cache and os.path.isfile(filename):
                print('Reuse cached file for variable('+str(i+1)+'/'+str(len(variable)) +')' +':'+variable[i])
            else:
                print('Download variable('+str(i+1)+'/'+str(len(variable)) +')' +':'+variable[i] )
                c.retrieve('reanalysis-era5-single-levels', cds_command, filename)
        return filename_list

    def download_temporary_files(self, ts: TimeSeries, use_cache: bool = False) -> Tuple[List[str], float, float]:
        raise NotImplementedError(f"Not implemented for {self.name}")

class GTSM(Product):

    def import_data(self, ts: TimeSeries, save_csv=True, save_nc=False, use_cache=False):
        """
        Extract times series of the nearest grid point (lon, lat) from
        GTSM water level and save it as netcdf.
        """
        if ts.variable == []:
            ts.variable = [
                "storm_surge_residual",
                "tidal_elevation",
                "total_water_level",
            ]
        filenames = self.__download_gtsm_from_cds(ts.start_time, ts.end_time, ts.variable, folder='cache')

        if not isinstance(filenames, list):
            filenames = [filenames]

        all_nc_files = []
        for filename in filenames:
            temppath = os.path.dirname(filename)
            # Unpack the tar.gz file.
            nc_files = subprocess.run(['tar', '-ztf', filename], stdout=subprocess.PIPE, check=True).stdout.decode('utf-8').split('\n')[0:-1]
            nc_files = sorted([ff.strip('\r') for ff in nc_files])
            subprocess.run(['tar', '-xzvf', filename, '--directory', temppath], stdout=subprocess.PIPE, check=True) # Extract tar file
            all_nc_files.extend([os.path.join(temppath, file) for file in nc_files])

        # Open multiple netCDF files as a single xarray dataset
        with xr.open_mfdataset(all_nc_files) as ds:
            # Calculate the distance to each station
            distances = np.sqrt((ds.station_x_coordinate - ts.lon)**2 + (ds.station_y_coordinate - ts.lat)**2)
            # Find the index of the nearest station
            nearest_station_index = distances.argmin().values
            # Extract the data for the nearest station
            ds = ds.isel(stations=nearest_station_index)
            lon_near=ds.stations.station_x_coordinate.values
            lat_near=ds.stations.station_y_coordinate.values
            ds = ds.drop_vars(['station_x_coordinate','station_y_coordinate', 'stations'], errors="ignore")
            df = aux_funcs.create_dataframe(self.name, ds, lon_near, lat_near, ts.datafile, ts.start_time, ts.end_time, save_csv)

        return df

    def __download_gtsm_from_cds(self,start_time, end_time, variable,  folder='cache') -> str:
        """
        Downloads GTSM model water level data from the Copernicus Climate Data Store for a
        given point and time period
        """
        import cdsapi
        filename = []
        filename_list = []
        start_time = pd.Timestamp(start_time)
        end_time = pd.Timestamp(end_time)
        c = cdsapi.Client()


        days = pd.date_range(start=start_time , end=end_time, freq='D')
        years = days.year
        years = years.unique()
        years = [str(year) for year in years]

        months = days.month
        months = months.unique()
        months = [f'{month:02}'  for month in months]

        # Create directory
        try:
            # Create target Directory
            os.mkdir(folder)
            print("Directory " , folder ,  " Created ")
        except FileExistsError:
            print("Directory " , folder ,  " already exists")

        for year in years:
            for var in variable:
                if var == 'tidal_elevation':
                    experiment = 'historical'
                else:
                    experiment = 'reanalysis'
                filename = f'{folder}/EC_GTSM_ERA5_{var}_{year}.tar.gz'
                cds_command = {
                    'experiment': experiment,
                    'format': 'tgz',
                    'variable': var,
                    'year': year, # 1950-2013
                    'month': months,
                    'temporal_aggregation' : '10_min',
                    #'model': 'CMCC-CM2-VHR4',

                }
                print('Download variable:',var, year)
                c.retrieve('sis-water-level-change-timeseries-cmip6', cds_command, filename)
                filename_list.append(filename)          
        return filename_list

    def download_temporary_files(self, ts: TimeSeries, use_cache: bool = False) -> Tuple[List[str], float, float]:
        raise NotImplementedError(f"Not implemented for {self.name}")
