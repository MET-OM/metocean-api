from __future__ import annotations
import os
from typing import TYPE_CHECKING
from datetime import datetime
import pandas as pd
import xarray as xr
from tqdm import tqdm

from .. import aux_funcs
from ..product import Product
from ..convention import Convention
if TYPE_CHECKING:
    from ts.ts_mod import TimeSeries  # Only imported for type checking


def find_product(name: str) -> Product:
    match name:
        case 'ECHOWAVE':
            return EchoWave(name)
    return None

class EchoWave(Product):

    @property
    def convention(self) -> Convention:
        return Convention.METEOROLOGICAL

    def import_data(self, ts: TimeSeries, save_csv=True, save_nc=False, use_cache=False):
        """
        Extract times series of  the nearest gird point (lon,lat) from
        ECHOWAVE wave hindcast and save it as netcdf.
        source: https://data.4tu.nl/datasets/f359cd0f-d135-416c-9118-e79dccba57b9/1
        """
        tempfiles, lon_near,lat_near = self.download_temporary_files(ts, use_cache)

        self.__remove_if_datafile_exists(ts.datafile)

        # merge temp files
        with xr.open_mfdataset(tempfiles) as ds:
            if save_nc:
                # Save the unaltered structure
                ds = ds.sel({"time": slice(ts.start_time, ts.end_time)})
                ds.to_netcdf(ts.datafile.replace(".csv", ".nc"), format='NETCDF3_64BIT') # add format to avoid *** AttributeError: NetCDF: String match to name in use
            df = self.__create_dataframe(
                ds=ds,
                lon_near=lon_near,
                lat_near=lat_near,
                outfile=ts.datafile,
                start_time=ts.start_time,
                end_time=ts.end_time,
                save_csv=save_csv
            )

        if not use_cache:
            # remove temp/cache files
            self.__clean_cache(tempfiles)
        return df

    def download_temporary_files(self, ts: TimeSeries, use_cache: bool = False):
        ts.variable = [ 'ucur', 'vcur', 'uwnd', 'vwnd', 'wlv', 'ice', 'hs', 'lm', 't02', 't01', 'fp', 'dir', 'spr', 'dp', 'phs0', 'phs1', 'phs2', 'ptp0', 'ptp1', 'ptp2', 'pdir0', 'pdir1']
        dates = self.__get_dates(start_date=ts.start_time, end_date=ts.end_time)

        tempfiles = aux_funcs.get_tempfiles(ts.product,ts.lon, ts.lat, dates)

        selection = None
        lon_near = None
        lat_near = None

        # extract point and create temp files
        for i in tqdm(range(len(dates))):
            url = self.__get_url_info(dates[i])

            if i == 0:
                selection, lon_near, lat_near = self.__get_near_coord(url, ts.lon, ts.lat)
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

        return tempfiles, lon_near, lat_near


    def __get_dates(self,start_date, end_date):
        return pd.date_range(start=datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y-%m') , end=datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y-%m'), freq='MS')


    def __create_dataframe(self,ds: xr.Dataset, lon_near, lat_near,outfile, start_time, end_time, save_csv=True):
        ds = ds.drop_vars(['longitude','latitude'], errors="ignore")
        return aux_funcs.create_dataframe(self.name, ds, lon_near, lat_near, outfile, start_time, end_time, save_csv)

    def __get_url_info(self,date) -> str:
        return f"https://opendap.4tu.nl/thredds/dodsC/data2/djht/f359cd0f-d135-416c-9118-e79dccba57b9/1/{date.strftime('%Y')}/TU-MREL_EU_ATL-2M_{date.strftime('%Y%m')}.nc"

    def __get_near_coord(self,infile, lon, lat):
        with xr.open_dataset(infile) as ds:
            ds_point = ds.sel(longitude=lon,latitude=lat,method='nearest')
            lon_near = ds_point.longitude.values
            lat_near = ds_point.latitude.values
            return {'longitude': lon_near, 'latitude': lat_near}, lon_near, lat_near

    def __clean_cache(self,tempfiles):
        for tmpfile in tempfiles:
            try:
                os.remove(tmpfile)
            except PermissionError:
                print(f"Skipping deletion of {tmpfile} due to PermissionError")

    def __remove_if_datafile_exists(self,datafile):
        if os.path.exists(datafile):
            os.remove(datafile)
