from __future__ import annotations
from typing import TYPE_CHECKING, Callable
import os
import xarray as xr
import pandas as pd
import numpy as np
from tqdm import tqdm
from .aux_funcs import (
    get_dates,
    get_url_info,
    get_tempfiles,
    get_near_coord,
    create_dataframe,
    read_commented_lines,
)

if TYPE_CHECKING:
    from ts.ts_mod import TimeSeries  # Only imported for type checking


def find_importer(product: str) -> Callable:
    match product:
        case "NORA3_wave_sub":
            return __nora3_wave_ts
        case "NORA3_wave":
            return __nora3_wave_ts
        case "NORA3_wind_sub":
            return __nora3_wind_ts
        case "NORAC_wave":
            return __norac_ts
        case "NORA3_wind_wave":
            return __nora3_combined_ts
        case "NORA3_stormsurge":
            return __nora3_stormsurge_ts
        case "NORA3_atm_sub":
            return __nora3_atm_ts
        case "NORA3_atm3hr_sub":
            return __nora3_atm3hr_ts
        case "NORKYST800":
            return __norkyst_800_ts
        case  "NorkystDA_surface":
            return __norkyst_da_surface_ts
        case "NorkystDA_zdepth":
            return __norkyst_da_zdepth_ts
        case 'NORA3_wave_spec':
            return __nora3_wave_spec
        case 'NORAC_wave_spec':
            return __norac_wave_spec

    if product.startswith("E39"):
        return __obs_e39

    return None

def __norac_ts(ts: TimeSeries, save_csv=False, save_nc=False, use_cache=False, download_only=False):
    """
    Extract times series of  the nearest gird point (lon,lat) from
    norac wave hindcast and save it as netcdf.
    """
    if ts.variable == []:
        ts.variable = [
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
    ts.variable.append("longitude")  # keep info of regular lon
    ts.variable.append("latitude")  # keep info of regular lat
    tempfiles, lon_near,lat_near = __download_temporary_files(ts, use_cache)
    if download_only:
        return tempfiles
    return __combine_temporary_files(ts,save_csv,save_nc,use_cache,tempfiles,lon_near,lat_near,ts.variable[:2],height=ts.height)


def __nora3_wave_ts(ts: TimeSeries, save_csv=False, save_nc=False, use_cache=False, download_only=False):
    if ts.variable == [] or ts.variable is None:
        ts.variable = [
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
    ts.variable.append("longitude")  # keep info of regular lon
    ts.variable.append("latitude")  # keep info of regular lat
    tempfiles, lon_near,lat_near = __download_temporary_files(ts, use_cache)
    if download_only:
        return tempfiles
    return __combine_temporary_files(ts,save_csv,save_nc,use_cache,tempfiles,lon_near,lat_near,ts.variable[:2],height=ts.height)

def __nora3_wind_ts(ts: TimeSeries, save_csv=False, save_nc=False, use_cache=False, download_only=False):
    """
    Extract times series of  the nearest gird point (lon,lat) from
    nora3 wind and wave hindcast and save it as netcdf.
    """
    if ts.variable == [] or ts.variable is None:
        ts.variable = ["wind_speed", "wind_direction"]

    ts.variable.append("longitude")  # keep info of regular lon
    ts.variable.append("latitude")  # keep info of regular lat
    tempfiles, lon_near,lat_near = __download_temporary_files(ts, use_cache)
    if download_only:
        return tempfiles
    return __combine_temporary_files(ts,save_csv,save_nc,use_cache,tempfiles,lon_near,lat_near,ts.variable[:2],height=ts.height)

def __nora3_atm_ts(ts: TimeSeries, save_csv=False, save_nc=False, use_cache=False, download_only=False):
    """
    Extract times series of  the nearest gird point (lon,lat) from
    nora3 atm. hindcast (parameteres exc. wind & waves) and save it as netcdf.
    """
    if ts.variable == []:
        ts.variable = [
            "air_pressure_at_sea_level",
            "air_temperature_2m",
            "relative_humidity_2m",
            "surface_net_longwave_radiation",
            "surface_net_shortwave_radiation",
            "precipitation_amount_hourly",
            "fog",
        ]
    ts.variable.append("longitude")  # keep info of regular lon
    ts.variable.append("latitude")  # keep info of regular lat

    tempfiles, lon_near,lat_near = __download_temporary_files(ts, use_cache)
    if download_only:
        return tempfiles

    return __combine_temporary_files(ts,save_csv,save_nc,use_cache,tempfiles,lon_near,lat_near,ts.variable[:2],height=ts.height)


def __nora3_atm3hr_ts(ts: TimeSeries, save_csv=False, save_nc=False, use_cache=False, download_only=False):
    """
    Extract times series of the nearest grid point (lon,lat) from
    nora3 atm. hindcast 3-hour files (parameters fex. wind & temperature) and save it as netcdf.
    """
    if ts.variable == [] or ts.variable is None:
        ts.variable = [
            "wind_speed",
            "wind_direction",
            "air_temperature",
            "relative_humidity",
            "density",
            "tke",
        ]
    ts.variable.append("longitude")  # keep info of regular lon
    ts.variable.append("latitude")  # keep info of regular lat
    tempfiles, lon_near,lat_near = __download_temporary_files(ts, use_cache)
    if download_only:
        return tempfiles
    return __combine_temporary_files(ts,save_csv,save_nc,use_cache,tempfiles,lon_near,lat_near,ts.variable[:-2],height=ts.height)


def __nora3_stormsurge_ts(ts: TimeSeries, save_csv=False, save_nc=False, use_cache=False, download_only=False):
    """
    Extract times series of  the nearest gird point (lon,lat) from
    nora3 sea level dataset and save it as netcdf.
    """
    ts.variable = ["zeta"]
    tempfiles, lon_near,lat_near = __download_temporary_files(ts, use_cache)
    if download_only:
        return tempfiles

    __remove_if_datafile_exists(ts.datafile)

    # merge temp files
    with xr.open_mfdataset(tempfiles) as ds:
        # TODO: This should be done when downloading the individual temporary files
        ds = ds.rename_dims({"ocean_time": "time"})
        ds = ds.rename_vars({"ocean_time": "time"})

        if save_nc:
            # Save the unaltered structure
            ds = ds.sel({"time": slice(ts.start_time, ts.end_time)})
            ds.to_netcdf(ts.datafile.replace(".csv", ".nc"))

        df = create_dataframe(
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
        __clean_cache(tempfiles)

    return df


def __norkyst_800_ts(ts: TimeSeries, save_csv=False, save_nc=False, use_cache=False, download_only=False):
    """
    Extract times series of  the nearest gird point (lon,lat) from
    Norkyst800.
    """
    if ts.variable == [] or ts.variable is None:
        ts.variable = ["salinity", "temperature", "u", "v", "zeta"]
    ts.variable.append("lon")  # keep info of regular lon
    ts.variable.append("lat")  # keep info of regular lat
    dates = get_dates(
        product=ts.product, start_date=ts.start_time, end_date=ts.end_time
    )

    tempfiles = get_tempfiles(ts.product, ts.lon, ts.lat, dates)

    selection = None
    lon_near = None
    lat_near = None

    # extract point and create temp files
    for i in range(len(dates)):
        url = get_url_info(ts.product, dates[i])

        # '2019-02-27' change to new model set up
        if i == 0 or dates[i].strftime("%Y-%m-%d %H:%M:%S") == "2019-02-27 00:00:00":
            selection, lon_near, lat_near = get_near_coord(
                url, ts.lon, ts.lat, ts.product
            )

        if use_cache and os.path.exists(tempfiles[i]):
            print(f"Found cached file {tempfiles[i]}. Using this instead")
        else:
            with xr.open_dataset(url) as dataset:
                # Reduce to the wanted variables and coordinates
                dataset = dataset[ts.variable]
                dataset = dataset.sel(selection).squeeze(drop=True)
                dataset.to_netcdf(tempfiles[i])

    if download_only:
        return tempfiles

    return __combine_temporary_files(ts,save_csv,save_nc,use_cache,tempfiles,lon_near,lat_near,ts.variable[:-2],height=ts.height)


def __nora3_combined_ts(ts: TimeSeries, save_csv = True,save_nc = False, use_cache =False, download_only=False):
    if download_only:
        raise ValueError(f"Download only not supported for product {ts.product}")
    ts.variable = ['hs','tp','fpI', 'tm1','tm2','tmp','Pdir','thq', 'hs_sea','tp_sea','thq_sea' ,'hs_swell','tp_swell','thq_swell']
    ts.product = 'NORA3_wave_sub'
    df_wave = __nora3_wave_ts(ts, save_csv,save_nc,use_cache)
    if save_csv:
        top_header_wave = read_commented_lines(ts.datafile)
    os.remove(ts.datafile)
    ts.variable = ["wind_speed", "wind_direction"]
    ts.product = "NORA3_wind_sub"
    df_wind = __nora3_wind_ts(ts, save_csv, save_nc, use_cache)
    if save_csv:
        top_header_wind = read_commented_lines(ts.datafile)
    os.remove(ts.datafile)

    # merge dataframes
    df = df_wind.join(df_wave)
    if save_csv:
        top_header = np.append(top_header_wave, top_header_wind)
        df.to_csv(ts.datafile, index_label="time")
        with open(ts.datafile, "r+", encoding="utf8") as f:
            content = f.read()
            f.seek(0, 0)
            for k in range(len(top_header) - 1):
                f.write(top_header[k].rstrip("\r\n") + "\n")
            f.write(top_header[-1].rstrip("\r\n") + "\n" + content)
        print("Data saved at: " + ts.datafile)
    if save_nc:
        df.to_netcdf(ts.datafile.replace(".csv", ".nc"))

    return df


def __norkyst_da_surface_ts(ts: TimeSeries, save_csv=False, save_nc=False, use_cache=False, download_only=False):
    """
    Extract times series of  the nearest gird point (lon,lat) from
    norkystDA surface dataset and save it as netcdf or csv.
    """
    ts.variable = ["u", "v", "zeta", "temp", "salt"]
    tempfiles, lon_near,lat_near = __download_temporary_files(ts, use_cache)
    if download_only:
        return tempfiles
    return __combine_temporary_files(ts,save_csv,save_nc,use_cache,tempfiles,lon_near,lat_near,ts.variable,height=ts.height)


def __norkyst_da_zdepth_ts(ts: TimeSeries, save_csv=False, save_nc=False, use_cache=False, download_only=False):
    """
    Extract times series of  the nearest gird point (lon,lat) from
    norkystDA surface dataset and save it as netcdf or csv.
    """
    ts.variable = ["u", "v", "zeta", "temp", "salt", "AKs"]
    tempfiles, lon_near,lat_near = __download_temporary_files(ts, use_cache)
    if download_only:
        return tempfiles
    return __combine_temporary_files(ts,save_csv,save_nc,use_cache,tempfiles,lon_near,lat_near,ts.variable,depth=ts.depth)


def __obs_e39(ts: TimeSeries, save_csv=False, save_nc=False, use_cache=False, download_only=False):
    """
    Extract times series of metocean E39 observations and save it as netcdf/csv.
    """
    if ts.variable == []:
        ts.variable = ["Hm0"]
    ts.variable.append("longitude")  # keep info of  lon
    ts.variable.append("latitude")  # keep info of  lat
    dates = get_dates(ts.product, ts.start_time, ts.end_time)
    tempfiles = get_tempfiles(ts.product, ts.lon, ts.lat, dates)

    # extract point and create temp files
    for i in range(len(dates)):
        if use_cache and os.path.exists(tempfiles[i]):
            print(f"Found cached file {tempfiles[i]}. Using this instead")
        else:
            url = get_url_info(product=ts.product, date=dates[i])
            with xr.open_dataset(url) as dataset:
                # Reduce to the wanted variables and coordinates
                dataset = dataset[ts.variable]
                dataset.to_netcdf(tempfiles[i])
                
    if download_only:
        return tempfiles

    __remove_if_datafile_exists(ts.datafile)

    # merge temp files
    with xr.open_mfdataset(tempfiles) as ds:

        ts.datafile.replace(
            ts.datafile.split("_")[-3],
            "lat" + str(np.round(ds.latitude.mean().values, 2)),
        )
        ts.datafile.replace(
            ts.datafile.split("_")[-4],
            "lon" + str(np.round(ds.longitude.mean().values, 2)),
        )
        if save_nc:
            # Save the unaltered structure
            ds = ds.sel({"time": slice(ts.start_time, ts.end_time)})
            ds.to_netcdf(ts.datafile.replace(".csv", ".nc"))

        df = create_dataframe(
            product=ts.product,
            ds=ds,
            lon_near=ds.longitude.mean().values,
            lat_near=ds.latitude.mean().values,
            outfile=ts.datafile,
            variable=ts.variable,
            start_time=ts.start_time,
            end_time=ts.end_time,
            save_csv=save_csv,
            height=ts.height,
        )
    if not use_cache:
        # remove temp/cache files
        __clean_cache(tempfiles)

    return df

def __nora3_wave_spec(ts: TimeSeries, save_csv=False, save_nc=False, use_cache=False, download_only=False):
    """
    Extract NORA3 wave spectra timeseries.
    """
    ts.variable = ['SPEC']
    ts.variable.append("longitude")  # keep info of regular lon
    ts.variable.append("latitude")  # keep info of regular lat
    dates = get_dates(ts.product, ts.start_time, ts.end_time)
    tempfiles = get_tempfiles(ts.product, ts.lon, ts.lat, dates)

    tempfiles, lon_near,lat_near = __download_temporary_files(ts, use_cache)
    if download_only:
        return tempfiles

    __remove_if_datafile_exists(ts.datafile)
    # merge temp files and create combined result
    with xr.open_mfdataset(tempfiles) as ds:
        da = ds["SPEC"]
        da.attrs["longitude"] = float(lon_near)
        da.attrs["latitude"] = float(lat_near)

    if save_csv:
        s = da.shape
        csv_data = {"time": da["time"].values.repeat(s[1]*s[2]),
            "frequency": np.tile(da["freq"].values.repeat(s[2]),s[0]),
            "direction": np.tile(da["direction"].values,s[0]*s[1]),
            "value": da.values.flatten()}
        csv_data = pd.DataFrame(csv_data,columns=["time","frequency","direction","value"])
        csv_data.to_csv(ts.datafile,index=False)

    if save_nc:
        # Save the unaltered structure
        ds = ds.sel({"time": slice(ts.start_time, ts.end_time)})
        ds.to_netcdf(ts.datafile.replace(".csv", ".nc"))

    return da

def __norac_wave_spec(ts: TimeSeries, save_csv=False, save_nc=False, use_cache=False, download_only=False) -> xr.DataArray:
    """
    Extract NORAC wave spectra timeseries.
    """
    ts.variable = ['efth']
    ts.variable.append("longitude")  # keep info of regular lon
    ts.variable.append("latitude")  # keep info of regular lat
    dates = get_dates(ts.product, ts.start_time, ts.end_time)
    tempfiles = get_tempfiles(ts.product, ts.lon, ts.lat, dates)

    tempfiles, lon_near,lat_near = __download_temporary_files(ts, use_cache)
    if download_only:
        return tempfiles

    __remove_if_datafile_exists(ts.datafile)
    # merge temp files and create combined result
    with xr.open_mfdataset(tempfiles) as ds:
        da = ds["efth"]
        da.attrs["longitude"] = float(lon_near)
        da.attrs["latitude"] = float(lat_near)

    if save_csv:
        s = da.shape
        csv_data = {"time": da["time"].values.repeat(s[1]*s[2]),
            "frequency": np.tile(da["frequency"].values.repeat(s[2]),s[0]),
            "direction": np.tile(da["direction"].values,s[0]*s[1]),
            "value": da.values.flatten()}
        csv_data = pd.DataFrame(csv_data,columns=["time","frequency","direction","value"])
        csv_data.to_csv(ts.datafile,index=False)

    if save_nc:
        # Save the unaltered structure
        ds = ds.sel({"time": slice(ts.start_time, ts.end_time)})
        ds.to_netcdf(ts.datafile.replace(".csv", ".nc"))

    return da

def __download_temporary_files(ts: TimeSeries, use_cache: bool = False):
    dates = get_dates(ts.product, ts.start_time, ts.end_time)

    tempfiles = get_tempfiles(ts.product, ts.lon, ts.lat, dates)
    selection = None
    lon_near = None
    lat_near = None

    # extract point and create temp files
    for i in tqdm(range(len(dates))):
        url = get_url_info(ts.product, dates[i])

        if i == 0:
            selection, lon_near, lat_near = get_near_coord(
                url, ts.lon, ts.lat, ts.product
            )
            tqdm.write(f'Nearest point to lat.={ts.lat:.3f},lon.={ts.lon:.3f} was found at lat.={lat_near:.3f},lon.={lon_near:.3f}')

        if use_cache and os.path.exists(tempfiles[i]):
            tqdm.write(f"Found cached file {tempfiles[i]}. Using this instead")
        else:
            with xr.open_dataset(url) as dataset:
                tqdm.write(f"Downloading {url}.")
                # Reduce to the wanted variables and coordinates
                dataset = dataset[ts.variable]
                dataset = dataset.sel(selection).squeeze(drop=True)
                __alter_temporary_file_if_needed(ts.product, dataset)
                dataset.to_netcdf(tempfiles[i])

    ts.lat_data = lat_near
    ts.lon_data = lon_near
    ts.tempfiles = tempfiles
    return tempfiles,lon_near, lat_near

def __alter_temporary_file_if_needed(product: str, ds: xr.Dataset):
    match product:
        case "NORA3_wind_sub":
            # The encoding of the fill value is not always correct in the netcdf files on the server
            for var_name in ds.variables:
                var = ds[var_name]
                if 'fill_value' in var.attrs:
                    var.encoding['_FillValue'] =  var.attrs['fill_value']


def __combine_temporary_files(ts: TimeSeries, save_csv, save_nc, use_cache,tempfiles, lon_near,lat_near,variables_to_flatten, **flatten_dims):
    __remove_if_datafile_exists(ts.datafile)
    # merge temp files
    with xr.open_mfdataset(tempfiles) as ds:
        if save_nc:
            # Save the unaltered structure
            ds = ds.sel({"time": slice(ts.start_time, ts.end_time)})
            ds.to_netcdf(ts.datafile.replace(".csv", ".nc"))

        df = create_dataframe(
            product=ts.product,
            ds=ds,
            lon_near=lon_near,
            lat_near=lat_near,
            outfile=ts.datafile,
            variable=variables_to_flatten,
            start_time=ts.start_time,
            end_time=ts.end_time,
            save_csv=save_csv,
            **flatten_dims
        )

    if not use_cache:
        # remove temp/cache files
        __clean_cache(tempfiles)

    return df

def __clean_cache(tempfiles):
    for tmpfile in tempfiles:
        try:
            os.remove(tmpfile)
        except PermissionError:
            print(f"Skipping deletion of {tmpfile} due to PermissionError")


def __remove_if_datafile_exists(datafile):
    if os.path.exists(datafile):
        os.remove(datafile)
