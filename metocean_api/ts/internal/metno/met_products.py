from __future__ import annotations
from typing import TYPE_CHECKING, override, Tuple,List
import os
from datetime import datetime
import xarray as xr
import pandas as pd
import numpy as np
import cartopy.crs as ccrs

from .met_product import MetProduct

from ..product import Product
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

    if name.startswith("E39"):
        return E39Observations(name)

    return None

class Nora3Wave(MetProduct):

    @override
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

    @override
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

    def _drop_variables(self):
        return ["projection_ob_tran", "longitude", "latitude", "rlat", "rlon"]


class Nora3WaveSub(Nora3Wave):

    @override
    def get_dates(self, start_date, end_date):
        return pd.date_range(
            start=datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y-%m"),
            end=datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y-%m"),
            freq="MS",
        )

    def _get_url_info(self, date: str):
        return "https://thredds.met.no/thredds/dodsC/nora3_subset_wave/wave_tser/" + date.strftime("%Y%m") + "_NORA3wave_sub_time_unlimited.nc"


class NORA3WindSub(MetProduct):

    @override
    def get_default_variables(self):
        return ["wind_speed", "wind_direction"]

    @override
    def get_dates(self, start_date, end_date):
        return pd.date_range(
            start=datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y-%m"),
            end=datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y-%m"),
            freq="MS",
        )

    @override
    def _get_url_info(self, date: str):
        return "https://thredds.met.no/thredds/dodsC/nora3_subset_atmos/wind_hourly/arome3kmwind_1hr_" + date.strftime("%Y%m") + ".nc"

    @override
    def _get_near_coord(self, url: str, lon: float, lat: float):
        with xr.open_dataset(url) as ds:
            x, y = aux_funcs.find_nearest_cart_coord(ds.longitude, ds.latitude, lon, lat)
            lon_near = ds.longitude.sel(y=y, x=x).values[0][0]
            lat_near = ds.latitude.sel(y=y, x=x).values[0][0]
            return {"x": x.values[0], "y": y.values[0]}, lon_near, lat_near

    @override
    def _alter_temporary_dataset_if_needed(self, dataset: xr.Dataset):
        for var_name in dataset.variables:
            # The encoding of the fill value is not always correct in the netcdf files on the server
            var = dataset[var_name]
            if "fill_value" in var.attrs:
                var.encoding["_FillValue"] = var.attrs["fill_value"]
        return dataset

    @override
    def _flatten_data_structure(self, ds: xr.Dataset, **flatten_dims):
        variables_to_flatten = ["wind_speed", "wind_direction"]
        height = self._get_values_for_dimension(ds, flatten_dims, "height")
        for i in range(len(height)):
            variable_flattened = [k + "_" + str(height[i]) + "m" for k in variables_to_flatten]
            ds[variable_flattened] = ds[variables_to_flatten].sel(height=height[i])

        drop_var = ["projection_lambert", "longitude", "latitude", "x", "y", "height"]
        drop_var.extend(variables_to_flatten)
        return ds.drop_vars(drop_var, errors="ignore")


class NORA3WindWaveCombined(MetProduct):

    @override
    def get_default_variables(self):
        raise NotImplementedError("This method should not be called")

    @override
    def _get_url_info(self, date: str):
        raise NotImplementedError("This method should not be called")

    @override
    def _get_near_coord(self, url: str, lon: float, lat: float):
        raise NotImplementedError("This method should not be called")

    @override
    def get_dates(self, start_date, end_date):
        raise NotImplementedError("This method should not be called")
    
    @override
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
    
    @override
    def download_temporary_files(self, ts: TimeSeries, use_cache: bool = False) -> Tuple[List[str], float, float]:
        raise NotImplementedError(f"Not implemented for product {self.name}")


class NORACWave(MetProduct):

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

    @override
    def get_dates(self, start_date, end_date):
        return pd.date_range(
            start=datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y-%m"),
            end=datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y-%m"),
            freq="MS",
        )

    @override
    def _get_url_info(self, date: str):
        return "https://thredds.met.no/thredds/dodsC/norac_wave/field/ww3." + date.strftime("%Y%m") + ".nc"

    @override
    def _get_near_coord(self, url: str, lon: float, lat: float):
        with xr.open_dataset(url) as ds:
            node_id = aux_funcs.distance_2points(ds.latitude.values, ds.longitude.values, lat, lon).argmin()
            lon_near = ds.longitude.sel(node=node_id).values
            lat_near = ds.latitude.sel(node=node_id).values
            return {"node": node_id}, lon_near, lat_near


class NORA3AtmSub(MetProduct):

    @override
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

    @override
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

    def _drop_variables(self):
        return ["projection_lambert", "longitude", "latitude", "x", "y"]


class NORA3Atm3hrSub(MetProduct):

    @override
    def get_default_variables(self):
        return [
            "wind_speed",
            "wind_direction",
            "air_temperature",
            "relative_humidity",
            "density",
            "tke",
        ]

    @override
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

    def _flatten_data_structure(self, ds: xr.Dataset, **flatten_dims):
        # TODO: Do this automatically. Just find the dimensions that are not time and flatten them
        variables_to_flatten = ["wind_speed", "wind_direction", "density", "tke", "air_temperature", "relative_humidity"]
        height = self._get_values_for_dimension(ds, flatten_dims, "height")
        for i in range(len(height)):
            variable_flattened = [k + "_" + str(height[i]) + "m" for k in variables_to_flatten]
            ds[variable_flattened] = ds[variables_to_flatten].sel(height=height[i])
        drop_var = ["projection_lambert", "longitude", "latitude", "x", "y", "height"]
        drop_var.extend(variables_to_flatten)
        return ds.drop_vars(drop_var, errors="ignore")


class NORA3StormSurge(MetProduct):

    @override
    def get_default_variables(self):
        return ["zeta"]

    @override
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

    @override
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

    @override
    def get_default_variables(self):
        return ["salinity", "temperature", "u", "v", "zeta"]

    @override
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

        return self._combine_temporary_files(ts, save_csv, save_nc, use_cache, tempfiles, lon_near, lat_near, height=ts.height)

    def _flatten_data_structure(self, ds: xr.Dataset, **flatten_dims):
        variables_to_flatten = ["salinity", "temperature", "u", "v"]
        depth = self._get_values_for_dimension(ds, flatten_dims, "depth")
        ds0 = ds
        if "depth" in ds["zeta"].dims:
            ds["zeta"] = ds.zeta.sel(depth=0)

        var_list = []
        for var_name in variables_to_flatten:
            # Check if 'depth' is not in the dimensions of the variable
            if "depth" in ds[var_name].dims:
                # Append variable name to the list
                var_list.append(var_name)

        for i in range(len(depth)):
            variable_flattened = [k + "_" + str(depth[i]) + "m" for k in var_list]
            ds[variable_flattened] = ds0[var_list].sel(depth=depth[i], method="nearest")

        ds = ds.drop_vars(var_list, errors="ignore")
        ds = ds.drop_vars(["lon", "lat", "depth"], errors="ignore")
        return ds.squeeze(drop=True)


class NorkystDASurface(MetProduct):

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

    def _flatten_data_structure(self, ds: xr.Dataset, **flatten_dims):
        crs = ccrs.Stereographic(central_latitude=90, central_longitude=70, true_scale_latitude=60, false_easting=0, false_northing=0)
        angle = aux_funcs.proj_rotation_angle(crs, ds)
        u, v = aux_funcs.rotate_vectors_tonorth(angle, ds["u"].values, ds["v"].values)
        spd, direction = aux_funcs.uv2spddir(u, v)
        ds["current_speed"] = xr.DataArray(spd, dims=("time"), attrs={"standard_name": "sea_water_speed", "units": "meter seconds-1"})
        ds["current_direction"] = xr.DataArray(
            direction, dims=("time"), attrs={"standard_name": "sea_water_velocity_from_direction", "units": "degrees"}
        )
        ds = ds.drop_vars(["u", "v"])

        drop_var = ["lon", "lat", "x", "y", "zdepth"]
        return ds.drop_vars(drop_var, errors="ignore")


class NorkystDAZdepth(MetProduct):

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

    @override
    def _flatten_data_structure(self, ds: xr.Dataset, **flatten_dims):
        depth = self._get_values_for_dimension(ds, flatten_dims, "depth")
        variables_to_flatten = ["u", "v", "temp", "salt", "AKs"]
        crs = ccrs.Stereographic(central_latitude=90, central_longitude=70, true_scale_latitude=60, false_easting=0, false_northing=0)
        angle = aux_funcs.proj_rotation_angle(crs, ds)

        # TODO: Should be simplified..
        for i in range(len(depth)):
            variable_flattened = [k + "_" + str(depth[i]) + "m" for k in variables_to_flatten]
            ds[variable_flattened] = ds[variables_to_flatten].sel(depth=depth[i])
            u, v = aux_funcs.rotate_vectors_tonorth(angle, ds["u_{}m".format(depth[i])].values, ds["v_{}m".format(depth[i])].values)
            spd, direction = aux_funcs.uv2spddir(u, v)
            ds[f"current_speed_{depth[i]}m"] = xr.DataArray(
                spd, dims=("time"), attrs={"standard_name": "sea_water_speed", "units": "meter seconds-1"}
            )
            ds[f"current_direction_{depth[i]}m"] = xr.DataArray(
                direction, dims=("time"), attrs={"standard_name": "sea_water_velocity_from_direction", "units": "degrees"}
            )
            ds = ds.drop_vars([f"u_{depth[i]}m", "v_{}m".format(depth[i])])

        drop_var = ["lon", "lat", "x", "y", "depth"]
        drop_var.extend(variables_to_flatten)
        return ds.drop_vars(drop_var, errors="ignore")


class NORA3WaveSpectrum(MetProduct):

    @override
    def get_default_variables(self):
        return ["SPEC"]

    @override
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

    @override
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

    @override
    def get_default_variables(self):
        return ["efth"]

    @override
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

    @override
    def get_default_variables(self):
        return ["Hm0"]

    @override
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
