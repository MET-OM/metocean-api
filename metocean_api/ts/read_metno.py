from __future__ import annotations
from pathlib import Path
import os
from typing import TYPE_CHECKING
import xarray as xr
import numpy as np
from .aux_funcs import get_date_list, get_url_info, get_near_coord, create_dataframe, check_datafile_exists, read_commented_lines
if TYPE_CHECKING:
    from .ts_mod import TimeSeries  # Only imported for type checking

def NORAC_ts(ts: TimeSeries, save_csv = False, save_nc = False, use_cache =False):
    """
    Extract times series of  the nearest gird point (lon,lat) from
    norac wave hindcast and save it as netcdf.
    """
    ts.variable.append('longitude') # keep info of regular lon
    ts.variable.append('latitude')  # keep info of regular lat
    date_list = get_date_list(product=ts.product, start_date=ts.start_time, end_date=ts.end_time)

    tempfiles = __tempfile_dir(ts.product,ts.lon, ts.lat, date_list,dirName="cache")

    # extract point and create temp files
    for i in range(len(date_list)):
        x_coor_str, _, infile = get_url_info(product=ts.product, date=date_list[i])

        if i==0:
            x_coor, _, lon_near, lat_near = get_near_coord(infile=infile, lon=ts.lon, lat=ts.lat, product=ts.product)

        if use_cache and os.path.exists(tempfiles[i]):
            print(f"Found cached file {tempfiles[i]}. Using this instead")
        else:
            with xr.open_dataset(infile) as dataset:
                # Reduce to the wanted variables and coordinates
                dataset = dataset[ts.variable]
                selection = {x_coor_str: x_coor}
                dataset = dataset.sel(selection).squeeze(drop=True)
                dataset.to_netcdf(tempfiles[i])

    check_datafile_exists(ts.datafile)
    # merge temp files
    with xr.open_mfdataset(tempfiles) as ds:
        # Save in csv format
        df = create_dataframe(product=ts.product,ds=ds, lon_near=lon_near, lat_near=lat_near, outfile=ts.datafile, variable=ts.variable[:2], start_time = ts.start_time, end_time = ts.end_time, save_csv=save_csv,save_nc = save_nc, height=ts.height)    

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


def NORA3_wind_wave_ts(ts: TimeSeries, save_csv = False, save_nc = False, use_cache =False):
    """
    Extract times series of  the nearest gird point (lon,lat) from
    nora3 wind and wave hindcast and save it as netcdf.
    """
    ts.variable.append('longitude') # keep info of regular lon
    ts.variable.append('latitude')  # keep info of regular lat
    date_list = get_date_list(product=ts.product, start_date=ts.start_time, end_date=ts.end_time)

    tempfiles = __tempfile_dir(ts.product,ts.lon, ts.lat, date_list,dirName="cache")

    # extract point and create temp files
    for i in range(len(date_list)):
        x_coor_str, y_coor_str, infile = get_url_info(product=ts.product, date=date_list[i])

        if i==0:
            # FIXME: Use the cache to find these values
            x_coor, y_coor, lon_near, lat_near = get_near_coord(infile=infile, lon=ts.lon, lat=ts.lat, product=ts.product)

        if use_cache and os.path.exists(tempfiles[i]):
            print(f"Found cached file {tempfiles[i]}. Using this instead")
        else:
            with xr.open_dataset(infile) as dataset:
                # Reduce to the wanted variables and coordinates
                dataset = dataset[ts.variable]
                selection = {x_coor_str: x_coor, y_coor_str: y_coor}
                dataset = dataset.sel(selection).squeeze(drop=True)
                # FIXME: product istedet for
                for var_name in ["wind_speed","wind_direction"]:
                    if var_name in dataset:
                        dataset[var_name].encoding['_FillValue'] =  dataset[var_name].attrs['fill_value']

                dataset.to_netcdf(tempfiles[i])

    check_datafile_exists(ts.datafile)
    #merge temp files
    with xr.open_mfdataset(tempfiles) as ds:
        #Save in csv format
        df = create_dataframe(product=ts.product,ds=ds, lon_near=lon_near, lat_near=lat_near, outfile=ts.datafile, variable=ts.variable[:2], start_time = ts.start_time, end_time = ts.end_time, save_csv=save_csv, save_nc = save_nc, height=ts.height)
        
    #remove temp/cache files
    if not use_cache:
        # remove temp/cache files
        __clean_cache(tempfiles)

    return df

def NORA3_atm_ts(ts: TimeSeries, save_csv = False, save_nc = False, use_cache =False):
    """
    Extract times series of  the nearest gird point (lon,lat) from
    nora3 atm. hindcast (parameteres exc. wind & waves) and save it as netcdf.
    """
    ts.variable.append('longitude') # keep info of regular lon
    ts.variable.append('latitude')  # keep info of regular lat
    date_list = get_date_list(product=ts.product, start_date=ts.start_time, end_date=ts.end_time)        

    tempfiles = __tempfile_dir(ts.product,ts.lon, ts.lat, date_list,dirName="cache")

    # extract point and create temp files
    for i in range(len(date_list)):
        x_coor_str, y_coor_str, infile = get_url_info(product=ts.product, date=date_list[i])

        if i==0:
            x_coor, y_coor, lon_near, lat_near = get_near_coord(infile=infile, lon=ts.lon, lat=ts.lat, product=ts.product)

        if use_cache and os.path.exists(tempfiles[i]):
            print(f"Found cached file {tempfiles[i]}. Using this instead")
        else:
            with xr.open_dataset(infile) as dataset:
                # Reduce to the wanted variables and coordinates
                dataset = dataset[ts.variable]
                selection = {x_coor_str: x_coor, y_coor_str: y_coor}
                dataset = dataset.sel(selection).squeeze(drop=True)
                dataset.to_netcdf(tempfiles[i])


    check_datafile_exists(ts.datafile)
    #merge temp files
    with xr.open_mfdataset(tempfiles) as ds:
        df = create_dataframe(product=ts.product,ds=ds, lon_near=lon_near, lat_near=lat_near, outfile=ts.datafile, variable=ts.variable[:2], start_time = ts.start_time, end_time = ts.end_time, save_csv=save_csv,save_nc = save_nc, height=ts.height)    
   
    if not use_cache:
        #remove temp/cache files
        __clean_cache(tempfiles)
    return df

def NORA3_atm3hr_ts(ts: TimeSeries, save_csv = False, save_nc = False, use_cache =False):
    """
    Extract times series of the nearest grid point (lon,lat) from
    nora3 atm. hindcast 3-hour files (parameters fex. wind & temperature) and save it as netcdf.
    """
    ts.variable.append('longitude') # keep info of regular lon
    ts.variable.append('latitude')  # keep info of regular lat
    date_list = get_date_list(product=ts.product, start_date=ts.start_time, end_date=ts.end_time)        

    tempfiles = __tempfile_dir(ts.product,ts.lon, ts.lat, date_list,dirName="cache")

    # extract point and create temp files
    for i in range(len(date_list)):
        x_coor_str, y_coor_str, infile = get_url_info(product=ts.product, date=date_list[i])

        if i==0:
            x_coor, y_coor, lon_near, lat_near = get_near_coord(infile=infile, lon=ts.lon, lat=ts.lat, product=ts.product)

        if use_cache and os.path.exists(tempfiles[i]):
            print(f"Found cached file {tempfiles[i]}. Using this instead")
        else:
            with xr.open_dataset(infile) as dataset:
                # Reduce to the wanted variables and coordinates
                dataset = dataset[ts.variable]
                selection = {x_coor_str: x_coor, y_coor_str: y_coor}
                dataset = dataset.sel(selection).squeeze(drop=True)
                dataset.to_netcdf(tempfiles[i])

    check_datafile_exists(ts.datafile)
    #merge temp files
    with xr.open_mfdataset(tempfiles) as ds:
        # # make density and tke in the same numer of dimensions as the other parameters
        ds['density0'] = xr.ones_like(ds['wind_speed'])
        ds['density0'].attrs['units'] = ds['density'].units
        ds['density0'].attrs['standard_name'] = ds['density'].standard_name
        ds['density0'].attrs['long_name'] = ds['density'].long_name
        ds['density0'][:,2] = ds['density'][:,0,].values # 150 m 
        ds = ds.drop_vars('density')
        ds = ds.rename_vars({'density0': 'density'})

        ds['tke0'] = xr.ones_like(ds['wind_speed'])
        ds['tke0'].attrs['units'] = ds['tke'].units
        ds['tke0'].attrs['standard_name'] = '-'
        ds['tke0'].attrs['long_name'] = ds['tke'].long_name
        ds['tke0'][:,2] = ds['tke'][:,0].values # 150 m  
        ds = ds.drop_vars('tke')
        ds = ds.rename_vars({'tke0': 'tke'})


        ds = ds.drop_vars('x')
        ds = ds.drop_vars('y')

        #Save in csv format  
        df = create_dataframe(product=ts.product,ds=ds, lon_near=lon_near, lat_near=lat_near, outfile=ts.datafile, variable=ts.variable[:-2], start_time = ts.start_time, end_time = ts.end_time, save_csv=save_csv,save_nc=save_nc, height=ts.height)
        
    if not use_cache:
    #remove temp/cache files
        __clean_cache(tempfiles)
    
    return df

def NORA3_stormsurge_ts(ts: TimeSeries, save_csv = False,save_nc = False, use_cache =False):
    """
    Extract times series of  the nearest gird point (lon,lat) from
    nora3 sea level dataset and save it as netcdf.
    """
    #ts.variable.append('lon_rho') # keep info of regular lon
    #ts.variable.append('lat_rho')  # keep info of regular lat
    date_list = get_date_list(product=ts.product, start_date=ts.start_time, end_date=ts.end_time)

    tempfiles = __tempfile_dir(ts.product,ts.lon, ts.lat, date_list,dirName="cache")

    # extract point and create temp files
    for i in range(len(date_list)):
        x_coor_str, y_coor_str, infile = get_url_info(product=ts.product, date=date_list[i])

        if i==0:
            x_coor, y_coor, lon_near, lat_near = get_near_coord(infile=infile, lon=ts.lon, lat=ts.lat, product=ts.product)        

        if use_cache and os.path.exists(tempfiles[i]):
            print(f"Found cached file {tempfiles[i]}. Using this instead")
        else:
            with xr.open_dataset(infile) as dataset:
                # Reduce to the wanted variables and coordinates
                dataset = dataset[ts.variable]
                selection = {x_coor_str: x_coor, y_coor_str: y_coor}
                dataset = dataset.sel(selection).squeeze(drop=True)
                dataset.to_netcdf(tempfiles[i])

    check_datafile_exists(ts.datafile)

    #merge temp files
    with xr.open_mfdataset(tempfiles) as ds:
        ds = ds.rename_dims({'ocean_time': 'time'})   
        ds = ds.rename_vars({'ocean_time': 'time'})   
        #Save in csv format    
        df = create_dataframe(product=ts.product,ds=ds, lon_near=lon_near, lat_near=lat_near, outfile=ts.datafile, variable=ts.variable, start_time = ts.start_time, end_time = ts.end_time, save_csv=save_csv,save_nc = save_nc, height=ts.height)    
    
    if not use_cache:
        #remove temp/cache files
        __clean_cache(tempfiles)
    
    return df

def NORKYST800_ts(ts: TimeSeries, save_csv = False, save_nc = False, use_cache = False):
    """
    Extract times series of  the nearest gird point (lon,lat) from
    Norkyst800.
    """
    ts.variable.append('lon') # keep info of regular lon
    ts.variable.append('lat')  # keep info of regular lat
    date_list = get_date_list(product=ts.product, start_date=ts.start_time, end_date=ts.end_time)        

    tempfiles = __tempfile_dir(ts.product,ts.lon, ts.lat, date_list,dirName="cache")

    # extract point and create temp files
    for i in range(len(date_list)):
        x_coor_str, y_coor_str, infile = get_url_info(product=ts.product, date=date_list[i])

        if i==0 or date_list[i].strftime('%Y-%m-%d %H:%M:%S') == '2019-02-27 00:00:00': # '2019-02-27' change to new model set up
            x_coor, y_coor, lon_near, lat_near = get_near_coord(infile=infile, lon=ts.lon, lat=ts.lat, product=ts.product)

        if use_cache and os.path.exists(tempfiles[i]):
            print(f"Found cached file {tempfiles[i]}. Using this instead")
        else:
            with xr.open_dataset(infile) as dataset:
                # Reduce to the wanted variables and coordinates
                dataset = dataset[ts.variable]
                selection = {x_coor_str: x_coor, y_coor_str: y_coor}
                dataset = dataset.sel(selection).squeeze(drop=True)
                dataset.to_netcdf(tempfiles[i])

    check_datafile_exists(ts.datafile)
    #merge temp files
    existing_files = [f for f in tempfiles if os.path.exists(f)]
    with xr.open_mfdataset(existing_files) as ds:
        #Save in csv format   
        df = create_dataframe(product=ts.product,ds=ds, lon_near=lon_near, lat_near=lat_near, outfile=ts.datafile, variable=ts.variable[:-2], start_time = ts.start_time, end_time = ts.end_time, save_csv=save_csv,save_nc = save_nc, height=ts.height)    
    
    if not use_cache:
        # remove temp/cache files
        __clean_cache(tempfiles)
    
    return df


def NORA3_combined_ts(ts: TimeSeries, save_csv = True,save_nc = False, use_cache =False):
    ts.variable = ['hs','tp','fpI', 'tm1','tm2','tmp','Pdir','thq', 'hs_sea','tp_sea','thq_sea' ,'hs_swell','tp_swell','thq_swell']
    ts.product = 'NORA3_wave_sub'
    df_wave = NORA3_wind_wave_ts(ts, save_csv,save_nc,use_cache)
    if save_csv:
        top_header_wave = read_commented_lines(ts.datafile)
    os.remove(ts.datafile)
    ts.variable = ['wind_speed','wind_direction']
    ts.product = 'NORA3_wind_sub' 
    df_wind = NORA3_wind_wave_ts(ts, save_csv,save_nc,use_cache)
    if save_csv:
        top_header_wind = read_commented_lines(ts.datafile)
    os.remove(ts.datafile)
    
    # merge dataframes
    df = df_wind.join(df_wave)
    if save_csv:
        top_header = np.append(top_header_wave,top_header_wind)
        df.to_csv(ts.datafile, index_label='time')
        with open(ts.datafile, 'r+', encoding="utf8") as f:
            content = f.read()
            f.seek(0, 0)
            for k in range(len(top_header)-1):                
                f.write(top_header[k].rstrip('\r\n') + '\n' )
            f.write(top_header[-1].rstrip('\r\n') + '\n' + content)
        print('Data saved at: ' +ts.datafile)
    if save_nc:
        df.to_netcdf(ts.datafile.replace('csv','nc'))

    return df

def NorkystDA_surface_ts(ts: TimeSeries, save_csv = False,save_nc = False, use_cache =False):
    """
    Extract times series of  the nearest gird point (lon,lat) from
    norkystDA surface dataset and save it as netcdf or csv.
    """
    #ts.variable.append('lon_rho') # keep info of regular lon
    #ts.variable.append('lat_rho')  # keep info of regular lat
    date_list = get_date_list(product=ts.product, start_date=ts.start_time, end_date=ts.end_time)        
    
    tempfiles = __tempfile_dir(ts.product,ts.lon, ts.lat, date_list,dirName="cache")

    # extract point and create temp files
    for i in range(len(date_list)):
        x_coor_str, y_coor_str, infile = get_url_info(product=ts.product, date=date_list[i])
             
        if i==0:
            x_coor, y_coor, lon_near, lat_near = get_near_coord(infile=infile, lon=ts.lon, lat=ts.lat, product=ts.product)  

        if use_cache and os.path.exists(tempfiles[i]):
            print(f"Found cached file {tempfiles[i]}. Using this instead")
        else:
            with xr.open_dataset(infile) as dataset:
                # Reduce to the wanted variables and coordinates
                dataset = dataset[ts.variable]
                selection = {x_coor_str: x_coor, y_coor_str: y_coor}
                dataset = dataset.sel(selection).squeeze(drop=True)
                dataset.to_netcdf(tempfiles[i])

    check_datafile_exists(ts.datafile)

    #merge temp files
    with xr.open_mfdataset(tempfiles) as ds:
        df = create_dataframe(product=ts.product,ds=ds, lon_near=lon_near, lat_near=lat_near, outfile=ts.datafile, variable=ts.variable, start_time = ts.start_time, end_time = ts.end_time, save_csv=save_csv,save_nc = save_nc, height=ts.height)    
    
    if not use_cache:
        #remove temp/cache files
        __clean_cache(tempfiles)

    return df

def NorkystDA_zdepth_ts(ts: TimeSeries, save_csv = False,save_nc = False, use_cache =False):
    """
    Extract times series of  the nearest gird point (lon,lat) from
    norkystDA surface dataset and save it as netcdf or csv.
    """
    #ts.variable.append('lon_rho') # keep info of regular lon
    #ts.variable.append('lat_rho')  # keep info of regular lat
    date_list = get_date_list(product=ts.product, start_date=ts.start_time, end_date=ts.end_time)        

    tempfiles = __tempfile_dir(ts.product,ts.lon, ts.lat, date_list,dirName="cache")

    # extract point and create temp files
    for i in range(len(date_list)):
        x_coor_str, y_coor_str, infile = get_url_info(product=ts.product, date=date_list[i])
             
        if i==0:
            x_coor, y_coor, lon_near, lat_near = get_near_coord(infile=infile, lon=ts.lon, lat=ts.lat, product=ts.product)

        if use_cache and os.path.exists(tempfiles[i]):
            print(f"Found cached file {tempfiles[i]}. Using this instead")
        else:
            with xr.open_dataset(infile) as dataset:
                # Reduce to the wanted variables and coordinates
                dataset = dataset[ts.variable]
                selection = {x_coor_str: x_coor, y_coor_str: y_coor}
                dataset = dataset.sel(selection).squeeze(drop=True)
                dataset.to_netcdf(tempfiles[i])
    
    check_datafile_exists(ts.datafile)
    
    #merge temp files
    with xr.open_mfdataset(tempfiles) as ds:
        df = create_dataframe(product=ts.product,ds=ds, lon_near=lon_near, lat_near=lat_near, outfile=ts.datafile, variable=ts.variable, start_time = ts.start_time, end_time = ts.end_time, save_csv=save_csv,save_nc = save_nc, height=ts.height, depth = ts.depth)    
    
    if not use_cache:
        #remove temp/cache files
        __clean_cache(tempfiles)

    return df

def __tempfile_dir(product,lon,lat, date_list,dirName):
    tempfile = [None] *len(date_list)
    # Create directory
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")
    
    for i in range(len(date_list)):
        tempfile[i] = str(Path(dirName+"/"+product+"_"+"lon"+str(lon)+"lat"+str(lat)+"_"+date_list.strftime('%Y%m%d')[i]+".nc"))

    return tempfile


def OBS_E39(ts: TimeSeries, save_csv = False, save_nc = False, use_cache =False):
    """
    Extract times series of metocean E39 observations and save it as netcdf/csv.
    """
    ts.variable.append('longitude') # keep info of  lon
    ts.variable.append('latitude')  # keep info of  lat
    date_list = get_date_list(product=ts.product, start_date=ts.start_time, end_date=ts.end_time) 
    tempfiles = __tempfile_dir(ts.product,ts.lon, ts.lat, date_list,dirName="cache")

    # extract point and create temp files
    for i in range(len(date_list)):
        _, _, infile = get_url_info(product=ts.product, date=date_list[i])

        if use_cache and os.path.exists(tempfiles[i]):
            print(f"Found cached file {tempfiles[i]}. Using this instead")
        else:
            with xr.open_dataset(infile) as dataset:
                # Reduce to the wanted variables and coordinates
                dataset = dataset[ts.variable]
                dataset.to_netcdf(tempfiles[i])

    check_datafile_exists(ts.datafile)

    #merge temp files
    with xr.open_mfdataset(tempfiles) as ds:
        #Save in csv format    
        ts.datafile.replace(ts.datafile.split('_')[-3],'lat'+str(np.round(ds.latitude.mean().values,2)))
        ts.datafile.replace(ts.datafile.split('_')[-4],'lon'+str(np.round(ds.longitude.mean().values,2)))
        df = create_dataframe(product=ts.product,ds=ds, lon_near=ds.longitude.mean().values, lat_near=ds.latitude.mean().values, outfile=ts.datafile, variable=ts.variable, start_time = ts.start_time, end_time = ts.end_time, save_csv=save_csv,save_nc = save_nc, height=ts.height)    
    if not use_cache:
        #remove temp/cache files
        __clean_cache(tempfiles)

    print('Data saved at: ' +ts.datafile)

    return df
