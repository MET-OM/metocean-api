from __future__ import annotations
from typing import TYPE_CHECKING, Tuple,List
import os
from datetime import datetime
import xarray as xr
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
from tqdm import tqdm

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
            self._remove_if_datafile_exists(ts.datafile)
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

        self._remove_if_datafile_exists(ts.datafile)

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
        date_range = pd.date_range(start=start_date, end=end_date, freq="h")
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
        return f"https://thredds.met.no/thredds/dodsC/nora3/{year:04}/{month:02}/{day:02}/{hour:02}/fc{year:04}{month:02}{day:02}{hour:02}_00{lead:1}_fp.nc"

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

    def is_same_forecast(self, date1, date2):
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
    
    def _correct_fluxes(tempfiles):
        fluxes = [
            'integral_of_surface_downward_sensible_heat_flux_wrt_time',
            'integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time',
            'integral_of_surface_downwelling_longwave_flux_in_air_wrt_time',
            'integral_of_toa_net_downward_shortwave_flux_wrt_time',
            'integral_of_surface_net_downward_shortwave_flux_wrt_time',
            'integral_of_toa_outgoing_longwave_flux_wrt_time',
            'integral_of_surface_net_downward_longwave_flux_wrt_time',
            'integral_of_surface_downward_latent_heat_evaporation_flux_wrt_time',
            'integral_of_surface_downward_latent_heat_sublimation_flux_wrt_time'
        ]
        print("Test")

    def download_temporary_files(self, ts: TimeSeries, use_cache: bool = False) -> Tuple[List[str], float, float]:
        if ts.variable == [] or ts.variable is None:
            ts.variable = self.get_default_variables()
        start_time = ts.start_time
        end_time = ts.end_time
        lat = ts.lat
        lon = ts.lon

        dates = self.get_dates(start_time, end_time)

        tempfiles = aux_funcs.get_tempfiles(self.name, lon, lat, dates)
        selection = None
        lon_near = None
        lat_near = None
        same_forecast = []

        # extract point and create temp files
        for i in tqdm(range(len(dates))):
            url = self._get_url_info(dates[i])

            if i == 0:
                selection, lon_near, lat_near = self._get_near_coord(url, lon, lat)
                tqdm.write(f"Nearest point to lat.={lat:.3f},lon.={lon:.3f} was found at lat.={lat_near:.3f},lon.={lon_near:.3f}")

            if use_cache and os.path.exists(tempfiles[i]):
                tqdm.write(f"Found cached file {tempfiles[i]}. Using this instead")
            else:
                with xr.open_dataset(url) as dataset:
                    tqdm.write(f"Downloading {url}.")
                    dataset.attrs["url"] = url
                    # Reduce to the wanted variables and coordinates
                    dataset = dataset[ts.variable]
                    dataset = dataset.sel(selection)
                    dimensions_to_squeeze = [dim for dim in dataset.dims if dim != 'time' and dataset.sizes[dim] == 1]
                    dataset = dataset.squeeze(dim=dimensions_to_squeeze, drop=True)
                    dataset = self._alter_temporary_dataset_if_needed(dataset)
                    dataset.to_netcdf(tempfiles[i])

            if self.is_same_forecast(dates[i], dates[i+1]):
                same_forecast.append(tempfiles[i])
            
            else:
                same_forecast.append(tempfiles[i])
                tqdm.write("Calculation of the hourly fluxes")
                print(same_forecast)

                same_forecast = []


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
