from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, List
import os
import tarfile
import pandas as pd
import xarray as xr
import numpy as np
from ..product import Product
from ..convention import Convention

from .. import aux_funcs

if TYPE_CHECKING:
    from ts.ts_mod import TimeSeries  # Only imported for type checking


def find_product(name: str) -> Product:
    match name:
        case "WA_NWShelf":
            return WA_NWShelf(name)
        case "PHY_NWShelf":
            return PHY_NWShelf(name)
    return None


class WA_NWShelf(Product):

    @property
    def convention(self) -> Convention:
        return Convention.METEOROLOGICAL

    def import_data(self, ts: TimeSeries, save_csv=True, save_nc=False, use_cache=False):
        """
        Extract times series of  the nearest grid point (lon,lat) from
        Copernicus reanalysis and save it as netcdf.
        """
        if ts.variable == []:
            ts.variable = [
                "VHM0",
                "VHM0_SW1",
                "VHM0_SW2",
                "VHM0_WW",
                "VMDR",
                "VMDR_SW1",
                "VMDR_SW2",
                "VMDR_WW",
                "VPED",
                "VSDX",
                "VSDY",
                "VTM01_SW1",
                "VTM01_SW2",
                "VTM01_WW",
                "VTM02",
                "VTM10",
                "VTPK",
            ]
        filenames = self.__download_nwshelf_from_cop(ts.start_time, ts.end_time, ts.lon, ts.lat, ts.variable, folder='cache')
        # Combine the data from the multiple files into a single dataframe
        df_res = None
        ds_res = None

        for filename in filenames:
            with xr.open_mfdataset(filename) as ds:
                # Rename the dimensions and variables to match the required format
                lon_near=ds.longitude.values[0]
                lat_near=ds.latitude.values[0]
                ds = ds.drop_vars(['longitude','latitude'], errors="ignore")
                df = aux_funcs.create_dataframe(self.name, ds, lon_near, lat_near, ts.datafile, ts.start_time, ts.end_time, save_csv=False)
                #df.drop(columns=['number', 'expver'], inplace=True, errors='ignore')
                if df_res is None:
                    df_res = df
                    ds_res = ds
                else:
                    df_res = df_res.join(df)
                    ds_res = ds_res.merge(ds, compat='override')

        if save_csv:
            # commenting because they have been deleted from the file, redefining here makes them 0
            # lon_near = ds.longitude.values[0]
            # lat_near = ds.latitude.values[0]
            header_lines =[f'#{ts.product};LONGITUDE:{lon_near:0.4f};LATITUDE:{lat_near:0.4f}']
            header_lines.append("#Variable_name;standard_name;long_name;units")
            var_names = ["time"]
            for name,vardata in ds_res.data_vars.items():
                varattr = vardata.attrs
                standard_name =varattr.get("standard_name", "-")
                long_name = varattr.get("long_name", "-")
                units = varattr.get("units", "-")
                header_lines.append("#" + name + ";" + standard_name + ";" + long_name + ";" + units)
                var_names.append(name)
            # Add column names last
            header_lines.append(",".join(var_names))
            header = "\n".join(header_lines) + "\n"

            with open(ts.datafile, "w", encoding="utf8", newline="") as f:
                f.write(header)
                df_res.to_csv(f, header=False, encoding=f.encoding, index_label="time")

        if save_nc:
            ds_res.to_netcdf(ts.datafile.replace('csv','nc'))

        return df_res

    def __download_nwshelf_from_cop(self,start_time, end_time, lon, lat, variable,  folder='cache', use_cache=True) -> str:
        """
        Downloads NWShelf wave data from the Copernicus Marine Data Store for a
        given point and time period
        """
        import copernicusmarine #Optional dependency
        start_time = pd.Timestamp(start_time)
        end_time = pd.Timestamp(end_time)

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
        filename_list = []
        usr = input("Enter your Copernicus Marine API username: ")
        pwd = input("Enter your Copernicus Marine API password: ")
        for i in range(len(variable)):
            filename = f'{folder}/NWShelf_'+"lon"+str(lon)+"lat"+str(lat)+"_"+days[0].strftime('%Y%m%d')+'_'+days[-1].strftime('%Y%m%d')+'_'+variable[i]+".nc"
            filename_list = np.append(filename_list,filename)


            if use_cache and os.path.isfile(filename):
                print('Reuse cached file for variable('+str(i+1)+'/'+str(len(variable)) +')' +':'+variable[i])
            else:
                print('Download variable('+str(i+1)+'/'+str(len(variable)) +')' +':'+variable[i] )
                copernicusmarine.subset(
                    dataset_id="MetO-NWS-WAV-RAN", 
                    variables=[variable[i]],
                    start_datetime=dates[0],
                    end_datetime=dates[-1],
                    minimum_longitude=lon-0.001,
                    maximum_longitude=lon+0.001,
                    minimum_latitude=lat-0.001,
                    maximum_latitude=lat+0.001,
                    output_filename=filename, 
                    username=usr,
                    password=pwd,
                    )

        return filename_list

    def download_temporary_files(self, ts: TimeSeries, use_cache: bool = False) -> Tuple[List[str], float, float]:
        raise NotImplementedError(f"Not implemented for {self.name}")
    

class PHY_NWShelf(Product):

    @property
    def convention(self) -> Convention:
        return Convention.METEOROLOGICAL

    def import_data(self, ts: TimeSeries, save_csv=True, save_nc=False, use_cache=False):
        """
        Extract times series of  the nearest grid point (lon,lat) from
        Copernicus reanalysis and save it as netcdf.
        """
        if ts.variable == []:
            ts.variable = [
                "zos",
                "uo",
                "vo",
            ]
        filenames = self.__download_nwshelf_from_cop(ts.start_time, ts.end_time, ts.lon, ts.lat, ts.variable, folder='cache')
        # Combine the data from the multiple files into a single dataframe
        df_res = None
        ds_res = None

        for filename in filenames:
            with xr.open_mfdataset(filename) as ds:
                # Rename the dimensions and variables to match the required format
                lon_near=ds.longitude.values[0]
                lat_near=ds.latitude.values[0]
                ds = ds.drop_vars(['longitude','latitude'], errors="ignore")
                df = aux_funcs.create_dataframe(self.name, ds, lon_near, lat_near, ts.datafile, ts.start_time, ts.end_time, save_csv=False)
                #df.drop(columns=['number', 'expver'], inplace=True, errors='ignore')
                if df_res is None:
                    df_res = df
                    ds_res = ds
                else:
                    df_res = df_res.join(df)
                    ds_res = ds_res.merge(ds, compat='override')

        if save_csv:
            # commenting because they have been deleted from the file, redefining here makes them 0
            # lon_near = ds.longitude.values[0]
            # lat_near = ds.latitude.values[0]
            header_lines =[f'#{ts.product};LONGITUDE:{lon_near:0.4f};LATITUDE:{lat_near:0.4f}']
            header_lines.append("#Variable_name;standard_name;long_name;units")
            var_names = ["time"]
            for name,vardata in ds_res.data_vars.items():
                varattr = vardata.attrs
                standard_name =varattr.get("standard_name", "-")
                long_name = varattr.get("long_name", "-")
                units = varattr.get("units", "-")
                header_lines.append("#" + name + ";" + standard_name + ";" + long_name + ";" + units)
                var_names.append(name)
            # Add column names last
            header_lines.append(",".join(var_names))
            header = "\n".join(header_lines) + "\n"

            with open(ts.datafile, "w", encoding="utf8", newline="") as f:
                f.write(header)
                df_res.to_csv(f, header=False, encoding=f.encoding, index_label="time")

        if save_nc:
            ds_res.to_netcdf(ts.datafile.replace('csv','nc'))

        return df_res

    def __download_nwshelf_from_cop(self,start_time, end_time, lon, lat, variable,  folder='cache', use_cache=True) -> str:
        """
        Downloads NWShelf physical data from the Copernicus Marine Data Store for a
        given point and time period
        """
        import copernicusmarine #Optional dependency
        start_time = pd.Timestamp(start_time)
        end_time = pd.Timestamp(end_time)

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
        filename_list = []
        usr = input("Enter your Copernicus Marine API username: ")
        pwd = input("Enter your Copernicus Marine API password: ")
        for i in range(len(variable)):
            if variable[i] == 'zos':
                dataset_id = "cmems_mod_nws_phy-ssh_my_7km-2D_PT1H-i"
            elif variable[i] == 'uo' or variable[i] == 'vo':
                dataset_id = "cmems_mod_nws_phy-uv_my_7km-2D_PT1H-i"
            else:
                raise NotImplementedError(f"Define product name for variable {variable[i]}")
            
            filename = f'{folder}/NWShelf_'+"lon"+str(lon)+"lat"+str(lat)+"_"+days[0].strftime('%Y%m%d')+'_'+days[-1].strftime('%Y%m%d')+'_'+variable[i]+".nc"
            filename_list = np.append(filename_list,filename)

            if use_cache and os.path.isfile(filename):
                print('Reuse cached file for variable('+str(i+1)+'/'+str(len(variable)) +')' +':'+variable[i])
            else:
                print('Download variable('+str(i+1)+'/'+str(len(variable)) +')' +':'+variable[i] )
                copernicusmarine.subset(
                    dataset_id=dataset_id, 
                    variables=[variable[i]],
                    start_datetime=dates[0],
                    end_datetime=dates[-1],
                    minimum_longitude=lon-0.001,
                    maximum_longitude=lon+0.001,
                    minimum_latitude=lat-0.001,
                    maximum_latitude=lat+0.001,
                    output_filename=filename,
                    username=usr,
                    password=pwd,
                    )

        return filename_list

    def download_temporary_files(self, ts: TimeSeries, use_cache: bool = False) -> Tuple[List[str], float, float]:
        raise NotImplementedError(f"Not implemented for {self.name}")
    
