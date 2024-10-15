from __future__ import annotations
from typing import TYPE_CHECKING
import os
import xarray as xr
import numpy as np
from .aux_funcs import (
    get_dates,
    get_url_info,
    get_tempfiles,
    get_near_coord,
    create_dataframe,
    read_commented_lines,
)

if TYPE_CHECKING:
    from .ts_mod import TimeSeries  # Only imported for type checking


def norac_ts(ts: TimeSeries, save_csv=False, save_nc=False, use_cache=False):
    """
    Extract times series of  the nearest gird point (lon,lat) from
    norac wave hindcast and save it as netcdf.
    """
    ts.variable.append("longitude")  # keep info of regular lon
    ts.variable.append("latitude")  # keep info of regular lat
    dates = get_dates(ts.product, ts.start_time, ts.end_time)

    tempfiles = get_tempfiles(ts.product, ts.lon, ts.lat, dates)

    selection = None
    lon_near = None
    lat_near = None

    # extract point and create temp files
    for i in range(len(dates)):
        url = get_url_info(ts.product, dates[i])

        if i == 0:
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

    __remove_if_datafile_exists(ts.datafile)
    # merge temp files and create combined result
    with xr.open_mfdataset(tempfiles) as ds:
        if save_nc:
            # Save the unaltered structure
            ds.to_netcdf(ts.datafile.replace(".csv", ".nc"))

        df = create_dataframe(
            product=ts.product,
            ds=ds,
            lon_near=lon_near,
            lat_near=lat_near,
            outfile=ts.datafile,
            variable=ts.variable[:2],
            start_time=ts.start_time,
            end_time=ts.end_time,
            save_csv=save_csv,
            height=ts.height,
        )

    if not use_cache:
        # remove temp/cache files
        __clean_cache(tempfiles)

    return df


def nora3_wind_wave_ts(ts: TimeSeries, save_csv=False, save_nc=False, use_cache=False):
    """
    Extract times series of  the nearest gird point (lon,lat) from
    nora3 wind and wave hindcast and save it as netcdf.
    """
    ts.variable.append("longitude")  # keep info of regular lon
    ts.variable.append("latitude")  # keep info of regular lat
    dates = get_dates(ts.product, ts.start_time, ts.end_time)

    tempfiles = get_tempfiles(ts.product, ts.lon, ts.lat, dates)

    selection = None
    lon_near = None
    lat_near = None

    # extract point and create temp files
    for i in range(len(dates)):
        url = get_url_info(ts.product, dates[i])

        if i == 0:
            # FIXME: Use the cache to find these values
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

    __remove_if_datafile_exists(ts.datafile)
    # merge temp files and create combined result
    with xr.open_mfdataset(tempfiles) as ds:
        if save_nc:
            # Save the unaltered structure
            ds.to_netcdf(ts.datafile.replace(".csv", ".nc"))

        df = create_dataframe(
            product=ts.product,
            ds=ds,
            lon_near=lon_near,
            lat_near=lat_near,
            outfile=ts.datafile,
            variable=ts.variable[:2],
            start_time=ts.start_time,
            end_time=ts.end_time,
            save_csv=save_csv,
            height=ts.height,
        )

    if not use_cache:
        # remove temp/cache files
        __clean_cache(tempfiles)

    return df


def nora3_atm_ts(ts: TimeSeries, save_csv=False, save_nc=False, use_cache=False):
    """
    Extract times series of  the nearest gird point (lon,lat) from
    nora3 atm. hindcast (parameteres exc. wind & waves) and save it as netcdf.
    """
    ts.variable.append("longitude")  # keep info of regular lon
    ts.variable.append("latitude")  # keep info of regular lat
    dates = get_dates(ts.product, ts.start_time, ts.end_time)

    tempfiles = get_tempfiles(ts.product, ts.lon, ts.lat, dates)
    selection = None
    lon_near = None
    lat_near = None

    # extract point and create temp files
    for i in range(len(dates)):
        url = get_url_info(ts.product, dates[i])

        if i == 0:
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

    __remove_if_datafile_exists(ts.datafile)
    # merge temp files and create combined result
    with xr.open_mfdataset(tempfiles) as ds:
        if save_nc:
            # Save the unaltered structure
            ds.to_netcdf(ts.datafile.replace(".csv", ".nc"))

        df = create_dataframe(
            product=ts.product,
            ds=ds,
            lon_near=lon_near,
            lat_near=lat_near,
            outfile=ts.datafile,
            variable=ts.variable[:2],
            start_time=ts.start_time,
            end_time=ts.end_time,
            save_csv=save_csv,
            height=ts.height,
        )

    if not use_cache:
        # remove temp/cache files
        __clean_cache(tempfiles)

    return df


def nora3_atm3hr_ts(ts: TimeSeries, save_csv=False, save_nc=False, use_cache=False):
    """
    Extract times series of the nearest grid point (lon,lat) from
    nora3 atm. hindcast 3-hour files (parameters fex. wind & temperature) and save it as netcdf.
    """
    ts.variable.append("longitude")  # keep info of regular lon
    ts.variable.append("latitude")  # keep info of regular lat
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

        if i == 0:
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

    __remove_if_datafile_exists(ts.datafile)
    # merge temp files
    with xr.open_mfdataset(tempfiles) as ds:
        if save_nc:
            # Save the unaltered structure
            ds.to_netcdf(ts.datafile.replace(".csv", ".nc"))

        # Save in csv format
        df = create_dataframe(
            product=ts.product,
            ds=ds,
            lon_near=lon_near,
            lat_near=lat_near,
            outfile=ts.datafile,
            variable=ts.variable[:-2],
            start_time=ts.start_time,
            end_time=ts.end_time,
            save_csv=save_csv,
            height=ts.height,
        )

    if not use_cache:
        # remove temp/cache files
        __clean_cache(tempfiles)

    return df


def nora3_stormsurge_ts(ts: TimeSeries, save_csv=False, save_nc=False, use_cache=False):
    """
    Extract times series of  the nearest gird point (lon,lat) from
    nora3 sea level dataset and save it as netcdf.
    """
    dates = get_dates(ts.product, ts.start_time, ts.end_time)

    tempfiles = get_tempfiles(ts.product, ts.lon, ts.lat, dates)

    selection = None
    lon_near = None
    lat_near = None

    # extract point and create temp files
    for i in range(len(dates)):
        url = get_url_info(ts.product, dates[i])

        if i == 0:
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

    __remove_if_datafile_exists(ts.datafile)

    # merge temp files
    with xr.open_mfdataset(tempfiles) as ds:
        ds = ds.rename_dims({"ocean_time": "time"})
        ds = ds.rename_vars({"ocean_time": "time"})

        if save_nc:
            # Save the unaltered structure
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


def norkyst_800_ts(ts: TimeSeries, save_csv=False, save_nc=False, use_cache=False):
    """
    Extract times series of  the nearest gird point (lon,lat) from
    Norkyst800.
    """
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

    __remove_if_datafile_exists(ts.datafile)
    # merge temp files
    with xr.open_mfdataset(tempfiles) as ds:
        if save_nc:
            # Save the unaltered structure
            ds.to_netcdf(ts.datafile.replace(".csv", ".nc"))
        df = create_dataframe(
            product=ts.product,
            ds=ds,
            lon_near=lon_near,
            lat_near=lat_near,
            outfile=ts.datafile,
            variable=ts.variable[:-2],
            start_time=ts.start_time,
            end_time=ts.end_time,
            save_csv=save_csv,
            height=ts.height,
        )

    if not use_cache:
        # remove temp/cache files
        __clean_cache(tempfiles)

    return df


def nora3_combined_ts(ts: TimeSeries, save_csv = True,save_nc = False, use_cache =False):
    ts.variable = ['hs','tp','fpI', 'tm1','tm2','tmp','Pdir','thq', 'hs_sea','tp_sea','thq_sea' ,'hs_swell','tp_swell','thq_swell']
    ts.product = 'NORA3_wave_sub'
    df_wave = nora3_wind_wave_ts(ts, save_csv,save_nc,use_cache)
    if save_csv:
        top_header_wave = read_commented_lines(ts.datafile)
    os.remove(ts.datafile)
    ts.variable = ["wind_speed", "wind_direction"]
    ts.product = "NORA3_wind_sub"
    df_wind = nora3_wind_wave_ts(ts, save_csv, save_nc, use_cache)
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


def norkyst_da_surface_ts(
    ts: TimeSeries, save_csv=False, save_nc=False, use_cache=False
):
    """
    Extract times series of  the nearest gird point (lon,lat) from
    norkystDA surface dataset and save it as netcdf or csv.
    """
    dates = get_dates(ts.product, ts.start_time, ts.end_time)

    tempfiles = get_tempfiles(ts.product, ts.lon, ts.lat, dates)

    selection = None
    lon_near = None
    lat_near = None

    # extract point and create temp files
    for i in range(len(dates)):
        url = get_url_info(ts.product, dates[i])

        if i == 0:
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

    __remove_if_datafile_exists(ts.datafile)

    # merge temp files
    with xr.open_mfdataset(tempfiles) as ds:
        if save_nc:
            # Save the unaltered structure
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


def norkyst_da_zdepth_ts(ts: TimeSeries, save_csv=False, save_nc=False, use_cache=False):
    """
    Extract times series of  the nearest gird point (lon,lat) from
    norkystDA surface dataset and save it as netcdf or csv.
    """
    dates = get_dates(ts.product, ts.start_time, ts.end_time)

    tempfiles = get_tempfiles(ts.product, ts.lon, ts.lat, dates)

    selection = None
    lon_near = None
    lat_near = None

    # extract point and create temp files
    for i in range(len(dates)):
        url = get_url_info(ts.product, dates[i])

        if i == 0:
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

    __remove_if_datafile_exists(ts.datafile)

    # merge temp files
    with xr.open_mfdataset(tempfiles) as ds:
        if save_nc:
            # Save the unaltered structure
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
            depth=ts.depth,
        )

    if not use_cache:
        # remove temp/cache files
        __clean_cache(tempfiles)

    return df


def obs_e39(ts: TimeSeries, save_csv=False, save_nc=False, use_cache=False):
    """
    Extract times series of metocean E39 observations and save it as netcdf/csv.
    """
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

    print("Data saved at: " + ts.datafile)

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
