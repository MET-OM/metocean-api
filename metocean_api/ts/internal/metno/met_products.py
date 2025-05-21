from __future__ import annotations
from typing import TYPE_CHECKING, Tuple,List
import os
from datetime import datetime, timezone
import xarray as xr
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
from tqdm import tqdm
from pathlib import Path
import metpy.calc as mpcalc
from metpy.units import units
import getpass

from .met_product import MetProduct

from ..product import Product
from ..convention import Convention
from .. import aux_funcs

if TYPE_CHECKING:
    from ts.ts_mod import TimeSeries  # Only imported for type checking


def find_product(name: str) -> Product:
    match name:
        case "NORA3_wave_sub":
            return Nora3WaveSub(name)
        case "NORA3_wave":
            return Nora3Wave(name)
        case "NORA3_wind_sub":
            return NORA3WindSub(name)
        case "NORAC_wave":
            return NORACWave(name)
        case "NORA3_wind_wave":
            return NORA3WindWaveCombined(name)
        case "NORA3_stormsurge":
            return NORA3StormSurge(name)
        case "NORA3_atm_sub":
            return NORA3AtmSub(name)
        case "NORA3_atm3hr_sub":
            return NORA3Atm3hrSub(name)
        case "NORKYST800":
            return Norkyst800(name)
        case "NorkystDA_surface":
            return NorkystDASurface(name)
        case "NorkystDA_zdepth":
            return NorkystDAZdepth(name)
        case "NORA3_wave_spec":
            return NORA3WaveSpectrum(name)
        case "NORAC_wave_spec":
            return NORACWaveSpectrum(name)
        case "NORA3_fpc":
            return NORA3fp(name)
        case "NORA3_":
            return NORA3_(name)
        case "NORA3_offshore_wind":
            return NORA3OffshoreWind(name)

    if name.startswith("E39"):
        return E39Observations(name)

    return None

class Nora3Wave(MetProduct):

    @property
    def convention(self) -> Convention:
        return Convention.OCEANIC

    def get_default_variables(self):
        return [
            "hs",
            "tp",
            "fpI",
            "tm1",
            "tm2",
            "tmp",
            "Pdir",
            "thq",
            "hs_sea",
            "tp_sea",
            "thq_sea",
            "hs_swell",
            "tp_swell",
            "thq_swell",
        ]

    def get_dates(self, start_date, end_date):
        return pd.date_range(start=start_date, end=end_date, freq="D")

    def _get_url_info(self, date: str):
        return (
            "https://thredds.met.no/thredds/dodsC/windsurfer/mywavewam3km_files/"
            + date.strftime("%Y")
            + "/"
            + date.strftime("%m")
            + "/"
            + date.strftime("%Y%m%d")
            + "_MyWam3km_hindcast.nc"
        )

    def _get_near_coord(self, url: str, lon: float, lat: float):
        with xr.open_dataset(url) as ds:
            rlon, rlat = aux_funcs.find_nearest_rot_coord(ds.longitude, ds.latitude, lon, lat)
            lon_near = ds.longitude.sel(rlat=rlat, rlon=rlon).values[0][0]
            lat_near = ds.latitude.sel(rlat=rlat, rlon=rlon).values[0][0]
            return {"rlon": rlon, "rlat": rlat}, lon_near, lat_near



class Nora3WaveSub(Nora3Wave):

    def get_dates(self, start_date, end_date):
        return pd.date_range(
            start=datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y-%m"),
            end=datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y-%m"),
            freq="MS",
        )

    def _get_url_info(self, date: str):
        return "https://thredds.met.no/thredds/dodsC/nora3_subset_wave/wave_tser/" + date.strftime("%Y%m") + "_NORA3wave_sub_time_unlimited.nc"


class NORA3WindSub(MetProduct):

    @property
    def convention(self) -> Convention:
        return Convention.METEOROLOGICAL

    def get_default_variables(self):
        return ["wind_speed", "wind_direction"]

    def get_dates(self, start_date, end_date):
        return pd.date_range(
            start=datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y-%m"),
            end=datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y-%m"),
            freq="MS",
        )

    def _get_url_info(self, date: str):
        return "https://thredds.met.no/thredds/dodsC/nora3_subset_atmos/wind_hourly_v2/arome3kmwind_1hr_" + date.strftime("%Y%m") + ".nc"

    def _get_near_coord(self, url: str, lon: float, lat: float):
        with xr.open_dataset(url) as ds:
            x, y = aux_funcs.find_nearest_cart_coord(ds.longitude, ds.latitude, lon, lat)
            lon_near = ds.longitude.sel(y=y, x=x).values[0][0]
            lat_near = ds.latitude.sel(y=y, x=x).values[0][0]
            return {"x": x.values[0], "y": y.values[0]}, lon_near, lat_near

    def _alter_temporary_dataset_if_needed(self, dataset: xr.Dataset):
        for var_name in dataset.variables:
            # The encoding of the fill value is not always correct in the netcdf files on the server
            var = dataset[var_name]
            if "fill_value" in var.attrs:
                var.encoding["_FillValue"] = var.attrs["fill_value"]
        return dataset

class NORA3WindWaveCombined(MetProduct):

    @property
    def convention(self) -> Convention:
        # This is a combined product, so we cannot determine the convention
        return Convention.NONE

    def get_default_variables(self):
        raise NotImplementedError("This method should not be called")

    def _get_url_info(self, date: str):
        raise NotImplementedError("This method should not be called")

    def _get_near_coord(self, url: str, lon: float, lat: float):
        raise NotImplementedError("This method should not be called")

    def get_dates(self, start_date, end_date):
        raise NotImplementedError("This method should not be called")

    def import_data(self, ts: TimeSeries, save_csv=True, save_nc=False, use_cache=False):
        product = ts.product

        wave = Nora3Wave("NORA3_wave_sub")
        wave_vars = wave.get_default_variables()
        ts.variable = wave_vars
        ts.product = wave.name
        wave_files, wave_lon_near, wave_lat_near = wave.download_temporary_files(ts, use_cache)
        lon_near = wave_lon_near
        lat_near = wave_lat_near

        wind = NORA3WindSub("NORA3_wind_sub")
        wind_vars = wind.get_default_variables()
        ts.variable = wind_vars
        ts.product = wind.name

        wind_files, wind_lon_near, wind_lat_near = wind.download_temporary_files(ts, use_cache)


        ts.variable = wave_vars + wind_vars
        ts.product = product

        tempfiles = wave_files + wind_files

        with xr.open_mfdataset(wave_files) as wave_values, xr.open_mfdataset(wind_files) as wind_values:
            # The latitude and longitude are not exactly the same for both datasets, so remove this from the wind dataset to be able merge them
            same_coords = wave_lon_near == wind_lon_near and wave_lat_near == wind_lat_near
            if not same_coords:
                print(
                    f"Coordinates for wave ({wave_lat_near},{wave_lon_near}) and wind ({wind_lat_near},{wind_lon_near}) are not the same. Using wave coordinates. "
                )
                wind_values = wind_values.drop_vars(["latitude", "longitude"], errors="ignore")
            aux_funcs.remove_if_datafile_exists(ts.datafile)
            # merge temp files
            if save_nc:
                ds = xr.merge([wave_values, wind_values])
                # Save the unaltered structure
                ds = ds.sel({"time": slice(ts.start_time, ts.end_time)})
                ds.to_netcdf(ts.datafile.replace(".csv", ".nc"))

            flatten_dims = {"height": ts.height}
            wave_ds = wave._flatten_data_structure(wave_values, **flatten_dims)
            wind_ds = wind._flatten_data_structure(wind_values, **flatten_dims)
            ds = xr.merge([wave_ds, wind_ds]).squeeze(drop=True)

            df = self.create_dataframe(
                ds=ds,
                lon_near=lon_near,
                lat_near=lat_near,
                outfile=ts.datafile,
                start_time=ts.start_time,
                end_time=ts.end_time,
                save_csv=save_csv,
                **flatten_dims
            )


        if not use_cache:
            # remove temp/cache files
            self._clean_cache(tempfiles)

        return df

    def download_temporary_files(self, ts: TimeSeries, use_cache: bool = False) -> Tuple[List[str], float, float]:
        raise NotImplementedError(f"Not implemented for product {self.name}")


class NORACWave(MetProduct):
    
    @property
    def convention(self) -> Convention:
        return Convention.METEOROLOGICAL

    def get_default_variables(self):
        return [
            "hs",
            "tp",
            "fpI",
            "t0m1",
            "t02",
            "t01",
            "dp",
            "dir",
            "phs0",
            "ptp0",
            "pdir0",
            "phs1",
            "ptp0",
            "pdir1",
        ]

    def get_dates(self, start_date, end_date):
        return pd.date_range(
            start=datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y-%m"),
            end=datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y-%m"),
            freq="MS",
        )

    def _get_url_info(self, date: str):
        return "https://thredds.met.no/thredds/dodsC/norac_wave/field/ww3." + date.strftime("%Y%m") + ".nc"

    def _get_near_coord(self, url: str, lon: float, lat: float):
        with xr.open_dataset(url) as ds:
            node_id = aux_funcs.distance_2points(ds.latitude.values, ds.longitude.values, lat, lon).argmin()
            lon_near = ds.longitude.sel(node=node_id).values
            lat_near = ds.latitude.sel(node=node_id).values
            return {"node": node_id}, lon_near, lat_near


class NORA3AtmSub(MetProduct):

    @property
    def convention(self) -> Convention:
        return Convention.METEOROLOGICAL

    def get_default_variables(self):
        return [
            "air_pressure_at_sea_level",
            "air_temperature_2m",
            "relative_humidity_2m",
            "surface_net_longwave_radiation",
            "surface_net_shortwave_radiation",
            "precipitation_amount_hourly",
            "fog",
        ]

    def get_dates(self, start_date, end_date):
        return pd.date_range(
            start=datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y-%m"),
            end=datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y-%m"),
            freq="MS",
        )

    def _get_url_info(self, date: str):
        return "https://thredds.met.no/thredds/dodsC/nora3_subset_atmos/atm_hourly/arome3km_1hr_" + date.strftime("%Y%m") + ".nc"

    def _get_near_coord(self, url: str, lon: float, lat: float):
        with xr.open_dataset(url) as ds:
            x, y = aux_funcs.find_nearest_cart_coord(ds.longitude, ds.latitude, lon, lat)
            lon_near = ds.longitude.sel(y=y, x=x).values[0][0]
            lat_near = ds.latitude.sel(y=y, x=x).values[0][0]
            return {"x": x.values[0], "y": y.values[0]}, lon_near, lat_near



class NORA3Atm3hrSub(MetProduct):

    @property
    def convention(self) -> Convention:
        return Convention.METEOROLOGICAL

    def get_default_variables(self):
        return [
            "wind_speed",
            "wind_direction",
            "air_temperature",
            "relative_humidity",
            "density",
            "tke",
            "sea_surface_temperature"
        ]

    def get_dates(self, start_date, end_date):
        return pd.date_range(
            start=datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y-%m"),
            end=datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y-%m"),
            freq="MS",
        )

    def _get_url_info(self, date: str):
        return "https://thredds.met.no/thredds/dodsC/nora3_subset_atmos/atm_3hourly/arome3km_3hr_" + date.strftime("%Y%m") + ".nc"

    def _get_near_coord(self, url: str, lon: float, lat: float):
        with xr.open_dataset(url) as ds:
            x, y = aux_funcs.find_nearest_cart_coord(ds.longitude, ds.latitude, lon, lat)
            lon_near = ds.longitude.sel(y=y, x=x).values[0][0]
            lat_near = ds.latitude.sel(y=y, x=x).values[0][0]
            return {"x": x.values[0], "y": y.values[0]}, lon_near, lat_near


class NORA3StormSurge(MetProduct):

    @property
    def convention(self) -> Convention:
        return Convention.NONE

    def get_default_variables(self):
        return ["zeta"]

    def get_dates(self, start_date, end_date):
        return pd.date_range(start=start_date, end=end_date, freq="YS")

    def _get_url_info(self, date: str):
        return "https://thredds.met.no/thredds/dodsC/stormrisk/zeta_nora3era5_N4_" + date.strftime("%Y") + ".nc"

    def _get_near_coord(self, url: str, lon: float, lat: float):
        with xr.open_dataset(url) as ds:
            eta_rho, xi_rho = aux_funcs.find_nearest_rho_coord(ds.lon_rho, ds.lat_rho, lon, lat)
            lon_near = ds.lon_rho.sel(eta_rho=eta_rho, xi_rho=xi_rho).values[0][0]
            lat_near = ds.lat_rho.sel(eta_rho=eta_rho, xi_rho=xi_rho).values[0][0]
            return {"eta_rho": eta_rho, "xi_rho": xi_rho}, lon_near, lat_near

    def _alter_temporary_dataset_if_needed(self, dataset: xr.Dataset):
        dataset = dataset.rename_dims({"ocean_time": "time"})
        return dataset.rename_vars({"ocean_time": "time"})

    def import_data(self, ts: TimeSeries, save_csv=False, save_nc=False, use_cache=False):
        """
        Extract times series of  the nearest gird point (lon,lat) from
        nora3 sea level dataset and save it as netcdf.
        """
        ts.variable = self.get_default_variables()
        tempfiles, lon_near, lat_near = self.download_temporary_files(ts, use_cache)

        aux_funcs.remove_if_datafile_exists(ts.datafile)

        # merge temp files
        with xr.open_mfdataset(tempfiles) as ds:
            if save_nc:
                # Save the unaltered structure
                ds = ds.sel({"time": slice(ts.start_time, ts.end_time)})
                ds.to_netcdf(ts.datafile.replace(".csv", ".nc"))

            df = self.create_dataframe(
                product=ts.product,
                ds=ds,
                lon_near=lon_near,
                lat_near=lat_near,
                outfile=ts.datafile,
                variable=ts.variable,
                start_time=ts.start_time,
                end_time=ts.end_time,
                save_csv=save_csv,
                height=ts.height,
            )

        if not use_cache:
            # remove temp/cache files
            self._clean_cache(tempfiles)

        return df

    def _drop_variables(self):
        return ["lon_rho", "lat_rho"]


class Norkyst800(MetProduct):

    @property
    def convention(self) -> Convention:
        return Convention.OCEANIC


    def get_default_variables(self):
        return ["salinity", "temperature", "u", "v", "zeta"]

    def get_dates(self, start_date, end_date):
        return pd.date_range(start=start_date, end=end_date, freq="D")

    def _get_url_info(self, date: str):
        if date >= pd.Timestamp("2016-09-14 00:00:00") and date <= pd.Timestamp("2019-02-26 00:00:00"):
            return "https://thredds.met.no/thredds/dodsC/sea/norkyst800mv0_1h/NorKyst-800m_ZDEPTHS_his.an." + date.strftime("%Y%m%d%H") + ".nc"
        elif date > pd.Timestamp("2019-02-26 00:00:00"):
            return "https://thredds.met.no/thredds/dodsC/fou-hi/norkyst800m-1h/NorKyst-800m_ZDEPTHS_his.an." + date.strftime("%Y%m%d%H") + ".nc"
        else:
            raise ValueError(f"Unhandled date {str(date)} for product {self.name}. Data only valid from 2016-09-14 onwards.")

    def _get_near_coord(self, url: str, lon: float, lat: float):
        with xr.open_dataset(url) as ds:
            x, y = aux_funcs.find_nearest_cart_coord(ds.lon, ds.lat, lon, lat)
            lon_near = ds.lon.sel(Y=y, X=x).values[0][0]
            lat_near = ds.lat.sel(Y=y, X=x).values[0][0]
            return {"X": x, "Y": y}, lon_near, lat_near

    def import_data(self, ts: TimeSeries, save_csv=True, save_nc=False, use_cache=False):
        """
        Extract times series of  the nearest gird point (lon,lat) from
        Norkyst800.
        """
        if ts.variable == [] or ts.variable is None:
            ts.variable = self.get_default_variables()
        ts.variable.append("lon")  # keep info of regular lon
        ts.variable.append("lat")  # keep info of regular lat
        dates = self.get_dates(start_date=ts.start_time, end_date=ts.end_time)

        tempfiles = aux_funcs.get_tempfiles(ts.product, ts.lon, ts.lat, dates)

        selection = None
        lon_near = None
        lat_near = None

        # extract point and create temp files
        for i in range(len(dates)):
            url = self._get_url_info(dates[i])

            # '2019-02-27' change to new model set up
            if i == 0 or dates[i].strftime("%Y-%m-%d %H:%M:%S") == "2019-02-27 00:00:00":
                selection, lon_near, lat_near = self._get_near_coord(url, ts.lon, ts.lat)

            if use_cache and os.path.exists(tempfiles[i]):
                print(f"Found cached file {tempfiles[i]}. Using this instead")
            else:
                with xr.open_dataset(url) as dataset:
                    # Reduce to the wanted variables and coordinates
                    dataset = dataset[ts.variable]
                    dataset = dataset.sel(selection).squeeze(drop=True)
                    dataset.to_netcdf(tempfiles[i])
                    
        ts.lat_data = lat_near
        ts.lon_data = lon_near

        return self._combine_temporary_files(ts, save_csv, save_nc, use_cache, tempfiles, lon_near, lat_near, height=ts.height)

    def _flatten_data_structure(self, ds: xr.Dataset, **flatten_dims):
        if "zeta" in ds.variables:
            # Just use the surface value
            if "depth" in ds["zeta"].dims:
                ds["zeta"] = ds.zeta.sel(depth=0)

        return super()._flatten_data_structure(ds, **flatten_dims)


class NorkystDASurface(MetProduct):

    @property
    def convention(self) -> Convention:
        return Convention.OCEANIC


    def get_default_variables(self):
        return ["u", "v", "zeta", "temp", "salt"]

    def get_dates(self, start_date, end_date):
        return pd.date_range(start=start_date, end=end_date, freq="D")

    def _get_url_info(self, date: str):
        return "https://thredds.met.no/thredds/dodsC/nora3_subset_ocean/surface/{}/ocean_surface_2_4km-{}.nc".format(
            date.strftime("%Y/%m"), date.strftime("%Y%m%d")
        )

    def _get_near_coord(self, url: str, lon: float, lat: float):
        with xr.open_dataset(url) as ds:
            x, y = aux_funcs.find_nearest_cart_coord(ds.lon, ds.lat, lon, lat)
            lon_near = ds.lon.sel(y=y, x=x).values[0][0]
            lat_near = ds.lat.sel(y=y, x=x).values[0][0]
            return {"x": x.values[0], "y": y.values[0]}, lon_near, lat_near

    def _alter_temporary_dataset_if_needed(self, dataset: xr.Dataset) -> xr.Dataset:
        # Convert u and v to current_speed and current_direction
        if "u" in dataset.variables and "v" in dataset.variables:
            crs = ccrs.Stereographic(central_latitude=90, central_longitude=70, true_scale_latitude=60, false_easting=0, false_northing=0)
            angle = aux_funcs.proj_rotation_angle(crs, dataset)
            u = dataset["u"]
            v = dataset["v"]
            u_rot, v_rot = aux_funcs.rotate_vectors_tonorth(angle, u.values, v.values)
            spd, direction = aux_funcs.uv2spddir(u_rot, v_rot)
            dataset["current_speed"] = xr.DataArray(
                spd, dims=("time"), attrs={"standard_name": "sea_water_speed", "units": "meter seconds-1"}
            )
            dataset["current_direction"] = xr.DataArray(
                direction, dims=("time"), attrs={"standard_name": "sea_water_velocity_from_direction", "units": "degrees"}
            )
        return dataset

    def _drop_variables(self):
        return ["u","v"]

class NorkystDAZdepth(MetProduct):

    @property
    def convention(self) -> Convention:
        return Convention.OCEANIC


    def get_default_variables(self):
        return ["u", "v", "zeta", "temp", "salt", "AKs"]

    def get_dates(self, start_date, end_date):
        return pd.date_range(start=start_date, end=end_date, freq="D")

    def _get_url_info(self, date: str):
        return "https://thredds.met.no/thredds/dodsC/nora3_subset_ocean/zdepth/{}/ocean_zdepth_2_4km-{}.nc".format(
            date.strftime("%Y/%m"), date.strftime("%Y%m%d")
        )

    def _get_near_coord(self, url: str, lon: float, lat: float):
        with xr.open_dataset(url) as ds:
            x, y = aux_funcs.find_nearest_cart_coord(ds.lon, ds.lat, lon, lat)
            lon_near = ds.lon.sel(y=y, x=x).values[0][0]
            lat_near = ds.lat.sel(y=y, x=x).values[0][0]
            return {"x": x.values[0], "y": y.values[0]}, lon_near, lat_near

    def _alter_temporary_dataset_if_needed(self, dataset: xr.Dataset) -> xr.Dataset:
        # Convert u and v to current_speed and current_direction
        if "u" in dataset.variables and "v" in dataset.variables:
            crs = ccrs.Stereographic(central_latitude=90, central_longitude=70, true_scale_latitude=60, false_easting=0, false_northing=0)
            angle = aux_funcs.proj_rotation_angle(crs, dataset)
            u = dataset["u"]
            v = dataset["v"]
            u_rot, v_rot = aux_funcs.rotate_vectors_tonorth(angle, u.values, v.values)
            spd, direction = aux_funcs.uv2spddir(u_rot, v_rot)
            dataset["current_speed"] = xr.DataArray(
                spd, dims=("time","depth"), attrs={"standard_name": "sea_water_speed", "units": "meter seconds-1"}
            )
            dataset["current_direction"] = xr.DataArray(
                direction, dims=("time","depth"), attrs={"standard_name": "sea_water_velocity_from_direction", "units": "degrees"}
            )
        return dataset

    def _drop_variables(self):
        return ["u","v"]


class NORA3WaveSpectrum(MetProduct):

    @property
    def convention(self) -> Convention:
        return Convention.OCEANIC

    def get_default_variables(self):
        return ["SPEC"]

    def get_dates(self, start_date, end_date):
        return pd.date_range(start=start_date, end=end_date, freq="D")

    def _get_url_info(self, date: str):
        return (
            "https://thredds.met.no/thredds/dodsC/windsurfer/mywavewam3km_spectra/"
            + date.strftime("%Y")
            + "/"
            + date.strftime("%m")
            + "/"
            + "SPC"
            + date.strftime("%Y%m%d")
            + "00.nc"
        )

    def _get_near_coord(self, url: str, lon: float, lat: float):
        with xr.open_dataset(url) as ds:
            station = aux_funcs.distance_2points(ds.latitude.values, ds.longitude.values, lat, lon).argmin()
            lon_near = ds.longitude.values[0, station]
            lat_near = ds.latitude.values[0, station]
            station += 1  # station labels are 1-indexed
            return {"x": station}, lon_near, lat_near

    def import_data(self, ts: TimeSeries, save_csv=True, save_nc=False, use_cache=False):
        """
        Extract NORA3 wave spectra timeseries.
        """
        ts.variable = self.get_default_variables()
        ts.variable.append("longitude")  # keep info of regular lon
        ts.variable.append("latitude")  # keep info of regular lat
        dates = self.get_dates(ts.start_time, ts.end_time)
        tempfiles = aux_funcs.get_tempfiles(ts.product, ts.lon, ts.lat, dates)

        tempfiles, lon_near, lat_near = self.download_temporary_files(ts, use_cache)

        if os.path.exists(ts.datafile):
            os.remove(ts.datafile)

        # merge temp files and create combined result
        with xr.open_mfdataset(tempfiles) as ds:
            da = ds["SPEC"]
            da.attrs["longitude"] = float(lon_near)
            da.attrs["latitude"] = float(lat_near)

        if save_csv:
            s = da.shape
            csv_data = {
                "time": da["time"].values.repeat(s[1] * s[2]),
                "frequency": np.tile(da["freq"].values.repeat(s[2]), s[0]),
                "direction": np.tile(da["direction"].values, s[0] * s[1]),
                "value": da.values.flatten(),
            }
            csv_data = pd.DataFrame(csv_data, columns=["time", "frequency", "direction", "value"])
            csv_data.to_csv(ts.datafile, index=False)

        if save_nc:
            # Save the unaltered structure
            ds = ds.sel({"time": slice(ts.start_time, ts.end_time)})
            ds.to_netcdf(ts.datafile.replace(".csv", ".nc"))

        return da


class NORACWaveSpectrum(MetProduct):

    @property
    def convention(self) -> Convention:
        return Convention.OCEANIC

    def get_default_variables(self):
        return ["efth"]

    def get_dates(self, start_date, end_date):
        return pd.date_range(
            start=datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y-%m"),
            end=datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y-%m"),
            freq="MS",
        )

    def _get_url_info(self, date: str):
        return "https://thredds.met.no/thredds/dodsC/norac_wave/spec/ww3_spec." + date.strftime("%Y%m") + ".nc"

    def _get_near_coord(self, url: str, lon: float, lat: float):
        with xr.open_dataset(url) as ds:
            station = aux_funcs.distance_2points(ds.latitude.values, ds.longitude.values, lat, lon).argmin()
            lon_near = ds.longitude.values[0, station]
            lat_near = ds.latitude.values[0, station]
            station += 1  # station labels are 1-indexed
            return {"station": station}, lon_near, lat_near

    def import_data(self, ts: TimeSeries, save_csv=True, save_nc=False, use_cache=False):
        """
        Extract NORAC wave spectra timeseries.
        """
        ts.variable = self.get_default_variables()
        ts.variable.append("longitude")  # keep info of regular lon
        ts.variable.append("latitude")  # keep info of regular lat
        dates = self.get_dates(ts.start_time, ts.end_time)
        tempfiles = aux_funcs.get_tempfiles(ts.product, ts.lon, ts.lat, dates)

        tempfiles, lon_near, lat_near = self.download_temporary_files(ts, use_cache)
        if os.path.exists(ts.datafile):
            os.remove(ts.datafile)

        # merge temp files and create combined result
        with xr.open_mfdataset(tempfiles) as ds:
            da = ds["efth"]
            da.attrs["longitude"] = float(lon_near)
            da.attrs["latitude"] = float(lat_near)

        if save_csv:
            s = da.shape
            csv_data = {
                "time": da["time"].values.repeat(s[1] * s[2]),
                "frequency": np.tile(da["frequency"].values.repeat(s[2]), s[0]),
                "direction": np.tile(da["direction"].values, s[0] * s[1]),
                "value": da.values.flatten(),
            }
            csv_data = pd.DataFrame(csv_data, columns=["time", "frequency", "direction", "value"])
            csv_data.to_csv(ts.datafile, index=False)

        if save_nc:
            # Save the unaltered structure
            ds = ds.sel({"time": slice(ts.start_time, ts.end_time)})
            ds.to_netcdf(ts.datafile.replace(".csv", ".nc"))

        return da


class E39Observations(MetProduct):

    @property
    def convention(self) -> Convention:
        return Convention.METEOROLOGICAL

    def get_default_variables(self):
        return ["Hm0"]

    def get_dates(self, start_date, end_date):
        return pd.date_range(start=start_date, end=end_date, freq="MS")

    def _get_url_info(self, date: str):
        product = self.name
        return "https://thredds.met.no/thredds/dodsC/obs/buoy-svv-e39/" + date.strftime("%Y/%m/%Y%m") + "_" + product + ".nc"

    def _get_near_coord(self, url: str, lon: float, lat: float):
        raise ValueError(f"Should not have been called for product {self.name}")

    def import_data(self, ts: TimeSeries, save_csv=True, save_nc=False, use_cache=False):
        """
        Extract times series of metocean E39 observations and save it as netcdf/csv.
        """
        if ts.variable == [] or ts.variable is None:
            ts.variable = self.get_default_variables()
        ts.variable.append("longitude")  # keep info of  lon
        ts.variable.append("latitude")  # keep info of  lat
        dates = self.get_dates(ts.start_time, ts.end_time)
        tempfiles = aux_funcs.get_tempfiles(ts.product, ts.lon, ts.lat, dates)

        # extract point and create temp files
        for i in range(len(dates)):
            if use_cache and os.path.exists(tempfiles[i]):
                print(f"Found cached file {tempfiles[i]}. Using this instead")
            else:
                url = self._get_url_info(date=dates[i])
                with xr.open_dataset(url) as dataset:
                    # Reduce to the wanted variables and coordinates
                    dataset = dataset[ts.variable]
                    dataset.to_netcdf(tempfiles[i])

        if os.path.exists(ts.datafile):
            os.remove(ts.datafile)

        # merge temp files
        with xr.open_mfdataset(tempfiles) as ds:
            lon_near = ds.longitude.mean().values
            lat_near = ds.latitude.mean().values
            ts.datafile.replace(
                ts.datafile.split("_")[-3],
                "lat" + str(np.round(lat_near, 2)),
            )
            ts.datafile.replace(
                ts.datafile.split("_")[-4],
                "lon" + str(np.round(lon_near, 2)),
            )
            if save_nc:
                # Save the unaltered structure
                ds = ds.sel({"time": slice(ts.start_time, ts.end_time)})
                ds.to_netcdf(ts.datafile.replace(".csv", ".nc"))

            df = self.create_dataframe(
                product=ts.product,
                ds=ds,
                lon_near=lon_near,
                lat_near=lat_near,
                outfile=ts.datafile,
                start_time=ts.start_time,
                end_time=ts.end_time,
                save_csv=save_csv,
                height=ts.height,
            )
        if not use_cache:
            # remove temp/cache files
            self._clean_cache(tempfiles)

        return df


class NORA3fp(MetProduct):

    @property
    def convention(self) -> Convention:
        return Convention.METEOROLOGICAL

    def get_default_variables(self):
        return [
            'longitude',
            'latitude',
            'air_temperature_0m',
            'surface_geopotential',
            'liquid_water_content_of_surface_snow',
            'downward_northward_momentum_flux_in_air',
            'downward_eastward_momentum_flux_in_air',
            'integral_of_toa_net_downward_shortwave_flux_wrt_time',
            'integral_of_surface_net_downward_shortwave_flux_wrt_time',
            'integral_of_toa_outgoing_longwave_flux_wrt_time',
            'integral_of_surface_net_downward_longwave_flux_wrt_time',
            'integral_of_surface_downward_latent_heat_evaporation_flux_wrt_time',
            'integral_of_surface_downward_latent_heat_sublimation_flux_wrt_time',
            'water_evaporation_amount',
            'surface_snow_sublimation_amount_acc',
            'integral_of_surface_downward_sensible_heat_flux_wrt_time',
            'integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time',
            'integral_of_surface_downwelling_longwave_flux_in_air_wrt_time',
            'rainfall_amount',
            'snowfall_amount',
            'graupelfall_amount_acc',
            'air_temperature_2m',
            'relative_humidity_2m',
            'specific_humidity_2m',
            'x_wind_10m',
            'y_wind_10m',
            'cloud_area_fraction',
            'x_wind_gust_10m',
            'y_wind_gust_10m',
            'air_temperature_max',
            'air_temperature_min',
            'convective_cloud_area_fraction',
            'high_type_cloud_area_fraction',
            'medium_type_cloud_area_fraction',
            'low_type_cloud_area_fraction',
            'atmosphere_boundary_layer_thickness',
            'hail_diagnostic',
            'graupelfall_amount',
                
            'air_temperature_pl',
            'cloud_area_fraction_pl',
            'geopotential_pl',
            'relative_humidity_pl',
            'upward_air_velocity_pl',
            'x_wind_pl',
            'y_wind_pl',
                
            'air_pressure_at_sea_level',
            'lwe_thickness_of_atmosphere_mass_content_of_water_vapor',
            'x_wind_z',
            'y_wind_z',
            'surface_air_pressure',
            'lifting_condensation_level',
            'atmosphere_level_of_free_convection',
            'atmosphere_level_of_neutral_buoyancy',
            'wind_direction',
            'wind_speed',
            'precipitation_amount_acc',
            'snowfall_amount_acc']

    def get_dates(self, start_date, end_date):
        start_date = pd.to_datetime(start_date) - pd.Timedelta(hours=6)
        end_date = pd.to_datetime(end_date) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        date_range = pd.date_range(start=start_date, end=end_date, freq="6h")
        return date_range

    def _get_url_info(self, date, lead_time):
        year = date.strftime("%Y")
        month = date.strftime("%m")
        day = date.strftime("%d")
        hour = date.strftime("%H")
        return f"https://thredds.met.no/thredds/dodsC/nora3/{year:04}/{month:02}/{day:02}/{hour:02}/fc{year:04}{month:02}{day:02}{hour:02}_00{lead_time:1}_fp.nc"
    

    def _get_near_coord(self, url: str, lon: float, lat: float):
        with xr.open_dataset(url) as ds:
            x, y = aux_funcs.find_nearest_cart_coord(ds.longitude, ds.latitude, lon, lat)
            lon_near = ds.longitude.sel(y=y, x=x).values[0][0]
            lat_near = ds.latitude.sel(y=y, x=x).values[0][0]
            return {"x": x.values[0], "y": y.values[0]}, lon_near, lat_near
    
    def _alter_temporary_dataset_if_needed(self, dataset: xr.Dataset):
        # Override this method in subclasses to alter the dataset before saving it to a temporary file
        # Renaming variables that does not follow the standard names should also be done here
        
        dataset = dataset.rename({'height2' : 'height', 'height3':'height_clouds', 'pressure0':'pressure_level'})
        dataset['wind_speed'] = np.sqrt(dataset['x_wind_z']**2 + dataset['y_wind_z']**2)
        dataset['wind_from_direction'] = (270 - np.arctan2(dataset['y_wind_z'], dataset['x_wind_z']) * 180 / np.pi) % 360


        air_temperature_0m = dataset['air_temperature_0m'].expand_dims({'standard_height': [0]})
        air_temperature_2m = dataset['air_temperature_2m'].expand_dims({'standard_height': [2]})
        relative_humidity_2m = dataset['relative_humidity_2m'].expand_dims({'standard_height': [2]})
        specific_humidity_2m = dataset['specific_humidity_2m'].expand_dims({'standard_height': [2]})
        x_wind_10m = dataset['x_wind_10m'].expand_dims({'standard_height': [10]})
        y_wind_10m = dataset['y_wind_10m'].expand_dims({'standard_height': [10]})
        x_wind_gust_10m = dataset['x_wind_gust_10m'].expand_dims({'standard_height': [10]})
        y_wind_gust_10m = dataset['y_wind_gust_10m'].expand_dims({'standard_height': [10]})

        air_temperature = xr.concat([air_temperature_0m, air_temperature_2m], dim='standard_height')
        relative_humidity = relative_humidity_2m
        specific_humidity = specific_humidity_2m
        x_wind = x_wind_10m
        y_wind = y_wind_10m
        x_wind_gust = x_wind_gust_10m
        y_wind_gust = y_wind_gust_10m

        dataset = dataset.drop_vars(['air_temperature_0m', 'air_temperature_2m', 'relative_humidity_2m', 'specific_humidity_2m',
                        'x_wind_10m', 'y_wind_10m', 'x_wind_gust_10m', 'y_wind_gust_10m'])

        dataset = xr.merge([dataset, air_temperature.rename('air_temperature'), relative_humidity.rename('relative_humidity'),
                    specific_humidity.rename('specific_humidity'), x_wind.rename('x_wind'), y_wind.rename('y_wind'),
                    x_wind_gust.rename('x_wind_gust'), y_wind_gust.rename('y_wind_gust')])
        
        dataset = dataset.drop_vars(['x', 'y'])
        
        return dataset

    def _is_same_forecast(self, date1, date2):
        def generate_modified_url(date):
            url = self._get_url_info(date)
            parts = url.split('/')
            filename = parts[-1]
            filename_parts = filename.split('_')
            filename_parts[1] = '003'
            new_filename = '_'.join(filename_parts)
            parts[-1] = new_filename
            return '/'.join(parts)

        # Generate the modified URLs for both dates
        new_url1 = generate_modified_url(date1)
        new_url2 = generate_modified_url(date2)

        # Compare the two URLs
        return new_url1 == new_url2
    
    def _process_fluxes(self, ds):

        working_dict = {
            'liquid_water_content_of_surface_snow': {
                'long_name': 'Instantaneous Liquid Water Content of Surface Snow',
                'standard_name': 'liquid_water_content_of_surface_snow',
                'units': 'kg m-2', 
                '_multiplier': 1,  # No conversion needed if already instantaneous
                '_rename': 'liquid_water_content_of_surface_snow'
            },
            'downward_northward_momentum_flux_in_air': {
                'long_name': 'Instantaneous Downward Northward Momentum Flux in Air',
                'standard_name': 'downward_northward_momentum_flux_in_air',
                'units': 'N m-2',  
                '_multiplier': 1/3600, #Needed to fit ERA5 values
                '_rename': 'downward_northward_momentum_flux_in_air'
            },
            'downward_eastward_momentum_flux_in_air': {
                'long_name': 'Instantaneous Downward Eastward Momentum Flux in Air',
                'standard_name': 'downward_eastward_momentum_flux_in_air',
                'units': 'N m-2',  
                '_multiplier': 1/3600, #Needed to fit ERA5 values
                '_rename': 'downward_eastward_momentum_flux_in_air'
            },
            'integral_of_toa_net_downward_shortwave_flux_wrt_time': {
                'long_name': 'Instantaneous TOA Net Downward Shortwave Flux',
                'standard_name': 'toa_net_downward_shortwave_flux',
                'units': 'W m-2',
                '_multiplier': 1/3600,
                '_rename': 'toa_net_downward_shortwave_flux'
            },
            'integral_of_surface_net_downward_shortwave_flux_wrt_time': {
                'long_name': 'Instantaneous Surface Net Downward Shortwave Flux',
                'standard_name': 'surface_net_downward_shortwave_flux',
                'units': 'W m-2',
                '_multiplier': 1/3600,
                '_rename': 'surface_net_downward_shortwave_flux'
            },
            'integral_of_toa_outgoing_longwave_flux_wrt_time': {
                'long_name': 'Instantaneous TOA Outgoing Longwave Flux',
                'standard_name': 'toa_outgoing_longwave_flux',
                'units': 'W m-2',
                '_multiplier': 1/3600,
                '_rename': 'toa_outgoing_longwave_flux'
            },
            'integral_of_surface_net_downward_longwave_flux_wrt_time': {
                'long_name': 'Instantaneous Surface Net Downward Longwave Flux',
                'standard_name': 'surface_net_downward_longwave_flux',
                'units': 'W m-2',
                '_multiplier': 1/3600,
                '_rename': 'surface_net_downward_longwave_flux'
            },
            'integral_of_surface_downward_latent_heat_evaporation_flux_wrt_time': {
                'long_name': 'Instantaneous Surface Downward Latent Heat Flux from Evaporation',
                'standard_name': 'surface_downward_latent_heat_evaporation_flux',
                'units': 'W m-2',
                '_multiplier': 1/3600,
                '_rename': 'surface_downward_latent_heat_evaporation_flux'
            },
            'integral_of_surface_downward_latent_heat_sublimation_flux_wrt_time': {
                'long_name': 'Instantaneous Surface Downward Latent Heat Flux from Sublimation',
                'standard_name': 'surface_downward_latent_heat_sublimation_flux',
                'units': 'W m-2',
                '_multiplier': 1/3600,
                '_rename': 'surface_downward_latent_heat_sublimation_flux'
            },
            'integral_of_surface_downward_sensible_heat_flux_wrt_time' : {
                'long_name' : 'Instantaneous Surface Downward Sensible Heat Flux in Air',
                'standard_name' : 'surface_downward_sensible_heat_flux',
                'units' : 'W m-2',
                '_multiplier' : 1/3600,
                '_rename' : 'surface_downward_sensible_heat_flux'
            },
            'integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time' : {
                'long_name' : 'Surface Downwelling Shortwave Flux',
                'standard_name' : 'surface_downwelling_shortwave_flux_in_air',
                'units' : 'W m-2',
                '_multiplier' : 1/3600,
                '_rename' : 'surface_downwelling_shortwave_flux_in_air'
            },
            'integral_of_surface_downwelling_longwave_flux_in_air_wrt_time' : {
                'long_name' : 'Surface Downwelling Longwave Flux',
                'standard_name' : 'surface_downwelling_longwave_flux_in_air',
                'units' : 'W m-2',
                '_multiplier' : 1/3600,
                '_rename' : 'surface_downwelling_longwave_flux_in_air'
            },
            'rainfall_amount' : {
                'long_name' : 'Amount of rain precipitation in the current time step',
                'standard_name' : 'rainfall_amount',
                'units' : 'kg m-2',
                '_multiplier' : 1,
                '_rename' : 'rainfall_amount'
            },
            'snowfall_amount' : {
                'long_name' : 'Amount of snow precipitation in the current time step',
                'standard_name' : 'snowfall_amount',
                'units' : 'kg m-2',
                '_multiplier' : 1,
                '_rename' : 'snowfall_amount'
            },
            'graupelfall_amount_acc' : {
                'long_name' : 'Amount of graupel fall in the current time step',
                'standard_name' : 'graupel_precipitation_amount',
                'units' : 'kg m-2',
                '_multiplier' : 1,
                '_rename' : 'graupel_precipitation_amount'
            },
            'precipitation_amount_acc' : {
                'long_name' : 'Amount of precipitation in the current time step',
                'standard_name' : 'precipitation_amount',
                'units' : 'kg m-2',
                '_multiplier' : 1,
                '_rename' : 'precipitation_amount'
            },
            'snowfall_amount_acc' : {
                'long_name' : 'Amount solid Precipitation (snow+graupel+hail) in the current time step',
                'standard_name' : 'solid_precipitation_amount',
                'units' : 'kg m-2',
                '_multiplier' : 1,
                '_rename' : 'solid_precipitation_amount'
            },
            'surface_snow_sublimation_amount_acc' : {
                'long_name' : 'Instantaneous Surface Snow Sublimation Mass per Unit Area',
                'standard_name' : 'surface_snow_sublimation_amount',
                'units' : 'kg m-2',
                '_multiplier' : 1,
                '_rename' : 'surface_snow_sublimation_amount' #None #Conflict fix
            },
            'water_evaporation_amount_acc': {
                'long_name': 'Accumulated Water Evaporation Mass per Unit Area',
                'standard_name': 'water_evaporation_amount',
                'units': 'kg m-2',
                '_multiplier': 1,
                '_rename': 'water_evaporation_amount'
            }

        }

        fluxes = [flux for flux in working_dict.keys() if flux in ds.variables.keys()]

        if not fluxes:
            return ds

        for flux in fluxes:
            attr = ds[flux].attrs
            

            ds[flux] = ds[flux].diff('time') * working_dict[flux]['_multiplier']
            for key in working_dict[flux].keys():
                if not key[0] == "_":
                    attr[key] = working_dict[flux][key]
            try:
                attr.pop('metno_name', None)
            except:
                pass
                
            ds[flux].attrs = attr
            
        rename_dict= { key : working_dict[key]['_rename'] for key in working_dict.keys() if key in fluxes and working_dict[key]['_rename'] is not None}
       
        return ds.rename_vars(rename_dict).isel(time=slice(1, None))



    def download_temporary_files(self, ts: TimeSeries, use_cache: bool = False) -> Tuple[List[str], float, float]:
        if ts.variable == [] or ts.variable is None:
            ts.variable = self.get_default_variables()
        start_time = ts.start_time
        end_time = ts.end_time
        lat = ts.lat
        lon = ts.lon

        dates = self.get_dates(start_time, end_time)

        tempfiles = aux_funcs.get_tempfiles(self.name, lon, lat, dates + pd.Timedelta(hours=4))
        selection = None
        lon_near = None
        lat_near = None

        # extract point and create temp files

        selection, lon_near, lat_near = self._get_near_coord(self._get_url_info(dates[0], 3), lon, lat)
        tqdm.write(f"Nearest point to lat.={lat:.3f},lon.={lon:.3f} was found at lat.={lat_near:.3f},lon.={lon_near:.3f}")

        header = "|{:^20}|{:^25}|{:^20}|{:^25}|".format(
            "Nb forecast",
            "Nb file to download",
            "Time per forecast",
            "Estimated time"
        )

        row = "|{:^20}|{:^25}|{:^20}|{:^25}|".format(
            len(tempfiles),
            len(tempfiles) * 7,
            "95 s/forecast",
            aux_funcs.format_seconds_to_dhms(len(tempfiles) * 95)
        )

        separator = "-" * len(header)

        tqdm.write("\n".join(["", separator, header, separator, row, separator, ""]))

        pbar = tqdm(total=len(dates), desc="Downloading NORA3 raw hindcast")
        for i, forecast in enumerate(dates):
            if use_cache and os.path.exists(tempfiles[i]):
                tqdm.write(f"Found cached file {tempfiles[i]}. Using this instead")
                pbar.update(1)
                continue
                
            tqdm.write(f"Fetching forecast data {forecast.strftime('%Y-%m-%d %H:00')}")
            forecast_files = []
            for lead_time in range(3, 10):
                url = self._get_url_info(forecast, lead_time)
                file = Path(f"cache/nora_{lon_near:.3f}_{lat_near:.3f}"+ url.split('/')[-1])


                if use_cache and os.path.exists(file):
                    tqdm.write(f"Found cached file {file}. Using this instead")
                    forecast_files.append(file)
                    continue


                with xr.open_dataset(url) as dataset:
                    tqdm.write(f"Downloading {url}.")
                    dataset.attrs["url"] = url
                    # Reduce to the wanted variables and coordinates
                    vars = set(ts.variable).intersection(set(dataset.variables.keys()))
                    dataset = dataset[vars]
                    dataset = dataset.sel(selection)
                    dimensions_to_squeeze = [dim for dim in dataset.dims if dim != 'time' and dataset.sizes[dim] == 1]
                    dataset = dataset.squeeze(dim=dimensions_to_squeeze, drop=True)
                    dataset.to_netcdf(file)
                    forecast_files.append(file)
            

            tqdm.write(f"Processing forecast data {forecast.strftime('%Y-%m-%d %H:00')}")
            with xr.open_mfdataset(forecast_files) as ds:
                ds = self._process_fluxes(ds)
                ds = self._alter_temporary_dataset_if_needed(ds)
                ds.to_netcdf(tempfiles[i])
                

            #Clean the tempory forecast files
            if not use_cache:
                self._clean_cache(forecast_files)
            
            pbar.update(1)
        
        ts.lat_data = lat_near
        ts.lon_data = lon_near
        return tempfiles, lon_near, lat_near


class NORA3_(MetProduct):

    @property
    def convention(self) -> Convention:
        return Convention.METEOROLOGICAL

    def get_default_variables(self):
        return [
            'p0',
            'ap',
            'b',
            'x',
            'y',
            'longitude',
            'latitude',
            'land_area_fraction',
            'specific_humidity_ml',
            'turbulent_kinetic_energy_ml',
            'cloud_area_fraction_ml',
            'toa_net_downward_shortwave_flux',
            'surface_downwelling_shortwave_flux_in_air',
            'toa_outgoing_longwave_flux',
            'surface_downwelling_longwave_flux_in_air',
            'atmosphere_boundary_layer_thickness',
            'pressure_departure',
            'surface_air_pressure',
            'air_temperature_ml',
            'surface_geopotential',
            'x_wind_ml',
            'y_wind_ml',
            'air_pressure_at_sea_level',
            'precipitation_amount_acc']

    def get_dates(self, start_date, end_date):
        date_range = pd.date_range(start=start_date, end=end_date, freq="3h")
        adjusted_dates = date_range - pd.Timedelta(hours=3)
        return adjusted_dates

    def _generate_time_info(self, dt : str):
        run_start_hours = [0, 6, 12, 18]

        # Find the closest preceding run start hour
        hour = dt.hour
        run_start = max([h for h in run_start_hours if h <= hour])

        # Calculate the file number
        file_number = 3 + (hour - run_start)

        return run_start, file_number

    def _get_url_info(self, date: str):
        year = date.strftime("%Y")
        month = date.strftime("%m")
        day = date.strftime("%d")
        hour, lead = self._generate_time_info(date)
        return f"https://thredds.met.no/thredds/dodsC/nora3/{year:04}/{month:02}/{day:02}/{hour:02}/fc{year:04}{month:02}{day:02}{hour:02}_00{lead:1}.nc"

    def _get_near_coord(self, url: str, lon: float, lat: float):
        with xr.open_dataset(url) as ds:
            x, y = aux_funcs.find_nearest_cart_coord(ds.longitude, ds.latitude, lon, lat)
            lon_near = ds.longitude.sel(y=y, x=x).values[0][0]
            lat_near = ds.latitude.sel(y=y, x=x).values[0][0]
            return {"x": x.values[0], "y": y.values[0]}, lon_near, lat_near
    
    def _alter_temporary_dataset_if_needed(self, dataset: xr.Dataset):
        # Override this method in subclasses to alter the dataset before saving it to a temporary file
        # Renaming variables that does not follow the standard names should also be done here
        
        dataset = dataset.drop_vars(['x', 'y'])
        
        return dataset

class NORA3OffshoreWind(MetProduct):
    """
    A class for handling wind data and Obukhov length over the NORA3 domain.

    This product includes:
    - Wind speed and direction at various heights (10m to 750m) with an hourly time step.
    - Obukhov length, friction velocity, surface pressure, air temperature, and humidity with an hourly time step.
    - Turbulent kinetic energy at specific heights with a 3-hourly time step.

    Original datasets:
    - https://thredds.met.no/thredds/projects/nora3.html
    - https://thredds.met.no/thredds/catalog/nora3_subset_atmos/atm_3hourly/catalog.html

    This work was done at SINTEF within the Horizon Europe project WILLOW.

    Example:
        product = 'NORA3_offshore_wind'
    """

    @property
    def convention(self) -> Convention:
        return Convention.METEOROLOGICAL

    def get_default_variables(self):
        raise NotImplementedError("This method should not be called")

    def _get_url_info(self, date: str):
        raise NotImplementedError("This method should not be called")

    def _get_near_coord(self, url: str, lon: float, lat: float):
        raise NotImplementedError("This method should not be called")

    def get_dates(self, start_date, end_date):
        raise NotImplementedError("This method should not be called")


    def calc_L(self, downward_northward_momentum_flux, downward_eastward_momentum_flux, specific_humidity, temperature, pressure, sensible_flux, vap_latent_flux, k=0.4, g=9.80665, cp=1004.7, Lv=2500800):
        """
        Calculate the Obukhov length (L), a measure of atmospheric stability. This function use the equation from Stull, An Introduction to Boundary Layer Meteorology (equation 5.7c, page 181)

        Parameters:
        - downward_northward_momentum_flux (array-like): Downward flux of northward momentum. Must have referenced unit in the Dataarray to be retrieved by metpy.
        - downward_eastward_momentum_flux (array-like): Downward flux of eastward momentum. Must have referenced unit in the Dataarray to be retrieved by metpy.
        - specific_humidity (array-like): Specific humidity of the air. Must have referenced unit in the Dataarray to be retrieved by metpy.
        - temperature (array-like): Temperature of the air. Must have referenced unit in the Dataarray to be retrieved by metpy.
        - pressure (array-like): Atmospheric pressure. Must have referenced unit in the Dataarray to be retrieved by metpy.
        - sensible_flux (array-like): Sensible heat flux. Must have referenced unit in the Dataarray to be retrieved by metpy.
        - vap_latent_flux (array-like): Latent heat flux due to vapor. Must have referenced unit in the Dataarray to be retrieved by metpy.
        - k (float, optional): Von Kármán constant. Default is 0.4.
        - g (float, optional): Acceleration due to gravity. Default is 9.80665 m/s².
        - cp (float, optional): Specific heat capacity of dry air at constant pressure. Default is 1004.7 J/(kg·K).
        - Lv (float, optional): Latent heat of vaporization. Default is 2500800 J/kg.

        Returns:
        - L (Quantity): Obukhov length with units of meters (m).

        Dependencies:
        - metpy.calc: For meteorological calculations.
        - metpy.units: For handling units.
        - numpy: For numerical operations.

        This function calculates the Obukhov length (L) using various atmospheric parameters. It ensures that all input values have the correct units, computes necessary intermediate values such as mixing ratio, virtual potential temperature, and density, and then calculates the Obukhov length.
        """
        attrs = downward_northward_momentum_flux.attrs

        # Ensure input values have the correct units
        downward_northward_momentum_flux = downward_northward_momentum_flux.metpy.quantify()
        downward_eastward_momentum_flux = downward_eastward_momentum_flux.metpy.quantify()
        specific_humidity = specific_humidity.metpy.quantify()
        temperature = temperature.metpy.quantify()
        pressure = pressure.metpy.quantify()
        sensible_flux = -sensible_flux.metpy.quantify()
        vap_latent_flux = -vap_latent_flux.metpy.quantify()
        cp = cp * units('joule/(kilogram*kelvin)')
        Lv = Lv * units('J/kg')
        g = g * units('m/s^2')


        # Calculate mixing ratio, virtual potential temperature, and density
        mixing_ratio = mpcalc.mixing_ratio_from_specific_humidity(specific_humidity)
        theta_v = mpcalc.virtual_potential_temperature(pressure, temperature, mixing_ratio)
        density = mpcalc.density(pressure, temperature, mixing_ratio)
        theta = mpcalc.potential_temperature(pressure, temperature)

        #Calculation of friction velocity
        tau = np.sqrt(downward_northward_momentum_flux**2 + downward_eastward_momentum_flux**2)

        # Calculate wtheta_s
        wtheta_s = (sensible_flux / (density * cp) * (1 + 0.6078 * specific_humidity)) + (0.6078 * theta * (vap_latent_flux / (density * Lv)))
        friction_velocity = np.sqrt( tau / density)

        # Calculate L
        L = -(friction_velocity**3 * theta_v) / (k * g * wtheta_s)

        L, friction_velocity = L.metpy.convert_units('m').metpy.dequantify(), friction_velocity.metpy.convert_units('m/s').metpy.dequantify()

        if 'units' in attrs:
            del attrs['units']

        attrs['long_name'] = "Atmospheric Stability Obukhov Length Calculated Using Stulls 2009 Formula"
        attrs['standard_name'] = 'atmosphere_obukhov_length'
        L.attrs.update(attrs)

        attrs['long_name'] = "Magnitude of the Friction Velocity in the Atmosphere"
        attrs['standard_name'] = 'magnitude_friction_velocity_in_air'
        friction_velocity.attrs.update(attrs)

        return L, friction_velocity

    def _haversine(self, lon1, lat1, lon2, lat2):
        import math
        # Convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Radius of Earth in kilometers. Use 3956 for miles. Determines return value units.
        return c * r

    def import_data(self, ts: TimeSeries, save_csv=True, save_nc=False, use_cache=False, max_retries = 5):
        retry_count = 0

        while retry_count < max_retries:
                
            try:

                product = ts.product

                print("\n     Gathering 'NORA3_fpc' data...\n")

                fpc = NORA3fp('NORA3_fpc')
                fpc_vars = fpc.get_default_variables()
                ts.variable = fpc_vars
                ts.product = fpc.name
                fpc_files, fpc_lon_near, fpc_lat_near = fpc.download_temporary_files(ts, use_cache)

                print("\n     Gathering 'NORA3_atm3hr_sub' data...\n")
                atm = NORA3Atm3hrSub("NORA3_atm3hr_sub")
                atm_vars = atm.get_default_variables()
                ts.variable = atm_vars
                ts.product = atm.name
                atm_files, atm_lon_near, atm_lat_near = atm.download_temporary_files(ts, use_cache)
                ts.variable = fpc_vars + atm_vars
                ts.product = product

                distance = self._haversine(fpc_lon_near, fpc_lat_near, atm_lon_near, atm_lat_near)
                print(f"The distance between 'NORA3_fpc' grid point ({fpc_lat_near:.4f},{fpc_lon_near:.4f}) and 'NORA3_atm3hr_sub' grid point ({atm_lat_near:.4f},{atm_lon_near:.4f}) is {distance:.3f}km")

                tempfiles = fpc_files + atm_files

                print("\n     Processing dataset...\n     (Can take some time for long time series)\n")

                with xr.open_mfdataset(fpc_files, parallel=False, engine="netcdf4") as ds_fpc, xr.open_mfdataset(atm_files, parallel=False, engine="netcdf4") as ds_atm, aux_funcs.Spinner():
                    ds_fpc = ds_fpc.load()
                    ds_atm = ds_atm.load()

                    #Backup the file history
                    ds_fpc_history = ds_fpc.attrs.get('history', [])
                    ds_atm_history = ds_atm.attrs.get('history', [])

                    ####Adding friction velocity and Obukhov Length
                    ds_fpc['atmosphere_obukhov_length'], ds_fpc['magnitude_friction_velocity_in_air'] = self.calc_L(
                        ds_fpc.downward_northward_momentum_flux_in_air.metpy.convert_to_base_units(),
                        ds_fpc.downward_eastward_momentum_flux_in_air.metpy.convert_to_base_units(),
                        ds_fpc.specific_humidity.sel(standard_height=2).metpy.convert_to_base_units(),
                        ds_fpc.air_temperature.sel(standard_height = 2).metpy.convert_to_base_units(),
                        ds_fpc.surface_air_pressure.metpy.convert_to_base_units(),
                        ds_fpc.surface_downward_sensible_heat_flux.metpy.convert_to_base_units(),
                        ds_fpc.surface_downward_latent_heat_evaporation_flux.metpy.convert_to_base_units()
                    )

                    ds_fpc = ds_fpc.drop_vars(['wind_speed', 'wind_from_direction'])


                    ####Update Wind Speed and Direction Distribution

                    ds_fpc['wind_speed'] = mpcalc.wind_speed(ds_fpc.x_wind_z, ds_fpc.y_wind_z).metpy.convert_units('m/s').metpy.dequantify()
                    ds_fpc['wind_from_direction'] = mpcalc.wind_direction(ds_fpc.x_wind_z, ds_fpc.y_wind_z).metpy.convert_units('degree').metpy.dequantify()

                    # Copy attributes from x_wind_z to wind_speed and wind_from_direction, preserving MetPy attributes
                    wind_speed_attrs = {**ds_fpc.x_wind_z.attrs, **ds_fpc['wind_speed'].attrs}
                    wind_from_direction_attrs = {**ds_fpc.x_wind_z.attrs, **ds_fpc['wind_from_direction'].attrs}

                    # Update the specific attributes for wind_speed
                    wind_speed_attrs['long_name'] = "Wind speed"
                    wind_speed_attrs['standard_name'] = "wind_speed"

                    # Update the specific attributes for wind_from_direction
                    wind_from_direction_attrs['long_name'] = "Wind from direction"
                    wind_from_direction_attrs['standard_name'] = "wind_from_direction"

                    # Assign the merged attributes back to the variables
                    ds_fpc['wind_speed'].attrs = wind_speed_attrs
                    ds_fpc['wind_from_direction'].attrs = wind_from_direction_attrs


                    #### Add the 10m wind speed to the profile

                    ws10 = mpcalc.wind_speed(ds_fpc.x_wind.sel(standard_height=10), ds_fpc.y_wind.sel(standard_height=10)).metpy.dequantify()
                    wd10 = mpcalc.wind_direction(ds_fpc.x_wind.sel(standard_height=10), ds_fpc.y_wind.sel(standard_height=10)).metpy.dequantify()
                    existing_heights = ds_fpc.coords['height'].values
                    new_heights = np.append(existing_heights, 10)
                    ds_fpc = ds_fpc.reindex(height=new_heights, fill_value=np.nan)
                    ds_fpc = ds_fpc.sortby('height')
                    ds_fpc.wind_speed.values[:, 0] = ws10.values
                    ds_fpc.wind_from_direction.values[:, 0] = wd10.values


                    #### Creation of the 2m parameters from the aggregated parameters

                    air_temperature_2m = ds_fpc.air_temperature.sel(standard_height=2)
                    relative_humidity_2m = ds_fpc.relative_humidity.sel(standard_height=2)
                    specific_humidity_2m = ds_fpc.specific_humidity.sel(standard_height=2)

                    ds_fpc['air_temperature_2m'] = air_temperature_2m
                    ds_fpc['air_temperature_2m'].attrs['long_name'] = 'Air temperature at 2 meters'
                    ds_fpc['air_temperature_2m'].attrs['standard_name'] = 'air_temperature'

                    ds_fpc['relative_humidity_2m'] = relative_humidity_2m
                    ds_fpc['relative_humidity_2m'].attrs['long_name'] = 'Relative humidity at 2 meters'
                    ds_fpc['relative_humidity_2m'].attrs['standard_name'] = 'relative_humidity'

                    ds_fpc['specific_humidity_2m'] = specific_humidity_2m
                    ds_fpc['specific_humidity_2m'].attrs['long_name'] = 'Specific humidity at 2 meters'
                    ds_fpc['specific_humidity_2m'].attrs['standard_name'] = 'specific_humidity'


                    ####Drop unwanted variables
                    time_attrs = ds_fpc.time.attrs
                    time = ds_fpc['time']

                    ds_fpc_variables = [
                        'wind_speed',
                        'wind_from_direction',
                        'atmosphere_obukhov_length',
                        'magnitude_friction_velocity_in_air',
                        'air_temperature_2m',
                        'relative_humidity_2m',
                        'specific_humidity_2m',
                        'atmosphere_boundary_layer_thickness',
                        'surface_air_pressure'
                        'time',
                        'height',
                        'longitude',
                        'latitude'
                    ]
                    ds_fpc_to_drop = [var for var in ds_fpc.variables.keys() if var not in ds_fpc_variables]
                    ds_fpc = ds_fpc.drop_vars(ds_fpc_to_drop)

                    if 'time' in ds_fpc.dims and 'time' not in ds_fpc.coords:
                        ds_fpc = ds_fpc.assign_coords(time=time)
                        ds_fpc['time'].attrs = time_attrs
                        #print('ok')
                    

                    #### Add Tke from atmospherique:
                    ds_atm_tke = ds_atm.tke.rename({'height' : 'height_tke'}).resample(time='1h').asfreq().drop_vars(['x','y']).reindex(time=ds_fpc.time, fill_value=np.nan)
                    ds_fpc['specific_turbulent_kinetic_energy_of_air'] = ds_atm_tke
                    ds_fpc['specific_turbulent_kinetic_energy_of_air'].attrs['standard_name'] = 'specific_turbulent_kinetic_energy_of_air'

                    ### Update general attributes

                    new_attrs = {
                        'Conventions': 'CF-1.6',
                        'institution': 'Norwegian Meteorological Institute, MET Norway',
                        'creator_url': 'https://www.met.no',
                        'source': 'NORA3 and NORA3 3-hourly atmospheric subset',
                        'title': 'NORA3 subset for offshore wind energy evaluation',
                        'min_time': pd.Timestamp(ds_fpc.time.values[0]).strftime('%Y-%m-%d %H:%M:%SZ'),
                        'max_time': pd.Timestamp(ds_fpc.time.values[-1]).strftime('%Y-%m-%d %H:%M:%SZ'),
                        'geospatial_lat_min': f"{ds_fpc.latitude.min().values:.3f}",
                        'geospatial_lat_max': f"{ds_fpc.latitude.max().values:.3f}",
                        'geospatial_lon_min': f"{ds_fpc.longitude.min().values:.3f}",
                        'geospatial_lon_max': f"{ds_fpc.longitude.max().values:.3f}",
                        'references': 'NORA3 (see DOI:10.1175/JAMC-D-21-0029.1 and DOI:10.1175/JAMC-D-22-0005.1)',
                        'keywords': "GCMDSK:Earth Science > Atmosphere > Atmospheric Boundary Layer, GCMDSK:Earth Science > Atmosphere > Atmospheric Turbulence, GCMDSK:Earth Science > Atmosphere > Wind, GCMDSK:Earth Science > Atmosphere > Temperature, GCMDSK:Earth Science > Atmosphere > Humidity",
                        'keywords_vocabulary': "GCMDSK",
                        'license': 'https://www.met.no/en/free-meteorological-data/Licensing-and-crediting',
                        'comment': 'None',
                        'summary': "This dataset is an aggregation of the NORA3 hindcast dataset and 3-hourly subset of atmospheric data to be used for offshore wind evaluation. It has been created using the metocean-api Python toolbox. This product creation workflow using the Norwegian Meteorological Institute dataset (see reference and creator fields) has been created by SINTEF Energy Research (Louis Pauchet, Valentin Chabaud).",
                        'DODS_EXTRA.Unlimited_Dimension': 'time',
                        'url': 'https://thredds.met.no/thredds/projects/nora3.html',
                        'history': f"NORA3 3-hourly atm subset history:\n{ds_atm_history}\nNORA3 hourly output history:\n{ds_fpc_history}\nCurrent dataset history:\n{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} - {getpass.getuser()} - metocean-api: NORA3OffshoreWind.import_data",
                        'creator_name': 'Birgitte Rugaard Furevik, Hilde Haakenstad',
                        'creator_type': 'person, person',
                        'creator_email': 'birgitterf@met.no, hildeh@met.no',
                        'creator_role': 'Technical contact, Investigator',
                        'creator_institution': 'Norwegian Meteorological Institute, Norwegian Meteorological Institute',
                    }

                    ds_fpc.attrs = new_attrs

                    if save_nc:
                        ds_fpc = ds_fpc.sel({"time": slice(ts.start_time, ts.end_time)})
                        aux_funcs.save_to_netcdf(ds_fpc, ts.datafile.replace(".csv", ".nc"))

                    flatten_dims = {"height": ts.height, "height_tke" : ts.height}
                    ds = fpc._flatten_data_structure(ds_fpc, **flatten_dims)

                    df = self.create_dataframe(
                        ds=ds,
                        lon_near=fpc_lon_near,
                        lat_near=fpc_lat_near,
                        outfile=ts.datafile,
                        start_time=ts.start_time,
                        end_time=ts.end_time,
                        save_csv=save_csv,
                        **flatten_dims
                    )


                if not use_cache:
                    # remove temp/cache files
                    self._clean_cache(tempfiles)

                return df
            
            except KeyError as e:
                # Handle error of a non complete or corrupted or reoponned netcdf file
                key = e.args[0]
                if isinstance(key, list) and len(key) > 1:
                    file_path = key[1][0]

                    # If the problematic file exist, remove it.
                    if file_path.endswith('.nc'):
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            print(f"Removed NetCDF file: {file_path}")
                        else:
                            print(f"NetCDF file not found: {file_path}")

                        retry_count += 1
                        print(f"Retrying... Attempt {retry_count}/{max_retries}")
                    else:
                        print(f"KeyError not related to a NetCDF file: {e}")
                        raise
                else:
                    print(f"KeyError not related to a file: {e}")
                    raise
            except Exception as e:
                print(f"An error occurred: {e}")
                raise

        raise Exception("Max retries reached. Operation failed.")

    def download_temporary_files(self, ts: TimeSeries, use_cache: bool = False) -> Tuple[List[str], float, float]:
        raise NotImplementedError(f"Not implemented for product {self.name}")
