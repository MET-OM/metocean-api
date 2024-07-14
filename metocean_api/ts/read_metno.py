from abc import ABC, abstractmethod

import xarray as xr
import pandas as pd
import numpy as np
import time
import os
from nco import Nco
from pathlib import Path

from .aux_funcs import *

def NORAC_ts(self, save_csv = False, save_nc = False, save_cache =False):
    """
    Extract times series of  the nearest gird point (lon,lat) from
    norac wave hindcast and save it as netcdf.
    """
    self.variable.append('longitude') # keep info of regular lon
    self.variable.append('latitude')  # keep info of regular lat
    date_list = get_date_list(product=self.product, start_date=self.start_time, end_date=self.end_time)        

    tempfile = tempfile_dir(self.product,self.lon, self.lat, date_list,dirName="cache")

    # extract point and create temp files
    for i in range(len(date_list)):
        x_coor_str, y_coor_str, infile = get_url_info(product=self.product, date=date_list[i])
             
        if i==0:
            x_coor, y_coor, lon_near, lat_near = get_near_coord(infile=infile, lon=self.lon, lat=self.lat, product=self.product)        

        opt = ['-O -v '+",".join(self.variable)+' -d '+x_coor_str+','+str(x_coor)]

        apply_nco(infile,tempfile[i],opt)

    check_datafile_exists(self.datafile)
    #merge temp files
    ds = xr.open_mfdataset(paths=tempfile[:])    
    #Save in csv format    
    df = create_dataframe(product=self.product,ds=ds, lon_near=lon_near, lat_near=lat_near, outfile=self.datafile, variable=self.variable[:2], start_time = self.start_time, end_time = self.end_time, save_csv=save_csv,save_nc = save_nc, height=self.height)    
    ds.close()
   #remove temp/cache files
    if save_cache == False:
        for i in range(len(date_list)):
            try:
                os.remove(tempfile[i])
            except PermissionError:
                print(f"Skipping deletion of {tempfile[i]} due to PermissionError")
    
    return df


def NORA3_wind_wave_ts(self, save_csv = False, save_nc = False, save_cache =False):
    """
    Extract times series of  the nearest gird point (lon,lat) from
    nora3 wind and wave hindcast and save it as netcdf.
    """
    self.variable.append('longitude') # keep info of regular lon
    self.variable.append('latitude')  # keep info of regular lat
    date_list = get_date_list(product=self.product, start_date=self.start_time, end_date=self.end_time)        

    tempfile = tempfile_dir(self.product,self.lon, self.lat, date_list,dirName="cache")

    # extract point and create temp files
    for i in range(len(date_list)):
        x_coor_str, y_coor_str, infile = get_url_info(product=self.product, date=date_list[i])
             
        if i==0:
            x_coor, y_coor, lon_near, lat_near = get_near_coord(infile=infile, lon=self.lon, lat=self.lat, product=self.product)        

        opt = ['-O -v '+",".join(self.variable)+' -d '+x_coor_str+','+str(x_coor.values[0])+' -d '+y_coor_str+','+str(y_coor.values[0])]

        apply_nco(infile,tempfile[i],opt)

    check_datafile_exists(self.datafile)
    #merge temp files
    ds = xr.open_mfdataset(paths=tempfile[:])    
    #Save in csv format    
    df = create_dataframe(product=self.product,ds=ds, lon_near=lon_near, lat_near=lat_near, outfile=self.datafile, variable=self.variable[:2], start_time = self.start_time, end_time = self.end_time, save_csv=save_csv, save_nc = save_nc, height=self.height)    
    ds.close()
   #remove temp/cache files
    if save_cache == False:
        for i in range(len(date_list)):
            try:
                os.remove(tempfile[i])
            except PermissionError:
                print(f"Skipping deletion of {tempfile[i]} due to PermissionError")
    
    return df

def NORA3_atm_ts(self, save_csv = False, save_nc = False, save_cache =False):
    """
    Extract times series of  the nearest gird point (lon,lat) from
    nora3 atm. hindcast (parameteres exc. wind & waves) and save it as netcdf.
    """
    self.variable.append('longitude') # keep info of regular lon
    self.variable.append('latitude')  # keep info of regular lat
    date_list = get_date_list(product=self.product, start_date=self.start_time, end_date=self.end_time)        

    tempfile = tempfile_dir(self.product,self.lon, self.lat, date_list,dirName="cache")

    # extract point and create temp files
    for i in range(len(date_list)):
        x_coor_str, y_coor_str, infile = get_url_info(product=self.product, date=date_list[i])
             
        if i==0:
            x_coor, y_coor, lon_near, lat_near = get_near_coord(infile=infile, lon=self.lon, lat=self.lat, product=self.product)        

        opt = ['-O -v '+",".join(self.variable)+' -d '+x_coor_str+','+str(x_coor.values[0])+' -d '+y_coor_str+','+str(y_coor.values[0])]

        
        apply_nco(infile,tempfile[i],opt)

    check_datafile_exists(self.datafile)
    #merge temp files
    ds = xr.open_mfdataset(paths=tempfile[:])    
    #Save in csv format    
    df = create_dataframe(product=self.product,ds=ds, lon_near=lon_near, lat_near=lat_near, outfile=self.datafile, variable=self.variable[:2], start_time = self.start_time, end_time = self.end_time, save_csv=save_csv,save_nc = save_nc, height=self.height)    
    ds.close()
   #remove temp/cache files
    if save_cache == False:
        for i in range(len(date_list)):
            try:
                os.remove(tempfile[i])
            except PermissionError:
                print(f"Skipping deletion of {tempfile[i]} due to PermissionError")
    
    return df

def NORA3_atm3hr_ts(self, save_csv = False, save_nc = False, save_cache =False):
    """
    Extract times series of the nearest grid point (lon,lat) from
    nora3 atm. hindcast 3-hour files (parameters fex. wind & temperature) and save it as netcdf.
    """
    self.variable.append('longitude') # keep info of regular lon
    self.variable.append('latitude')  # keep info of regular lat
    date_list = get_date_list(product=self.product, start_date=self.start_time, end_date=self.end_time)        

    tempfile = tempfile_dir(self.product,self.lon, self.lat, date_list,dirName="cache")

    # extract point and create temp files
    for i in range(len(date_list)):
        x_coor_str, y_coor_str, infile = get_url_info(product=self.product, date=date_list[i])
             
        if i==0:
            x_coor, y_coor, lon_near, lat_near = get_near_coord(infile=infile, lon=self.lon, lat=self.lat, product=self.product)        

        opt = ['-O -v '+",".join(self.variable)+' -d '+x_coor_str+','+str(x_coor.values[0])+' -d '+y_coor_str+','+str(y_coor.values[0])]
                
        apply_nco(infile,tempfile[i],opt)

    check_datafile_exists(self.datafile)
    #merge temp files
    ds = xr.open_mfdataset(paths=tempfile[:])

    # make density and tke in the same numer of dimensions as the other parameteres    
    ds['density0'] = xr.ones_like(ds['wind_speed'])
    ds['density0'].attrs['units'] = ds['density'].units
    ds['density0'].attrs['standard_name'] = ds['density'].standard_name
    ds['density0'].attrs['long_name'] = ds['density'].long_name
    ds['density0'][:,2,:,:] = ds['density'][:,0,:,:].values # 150 m 
    ds = ds.drop_vars('density')
    ds = ds.rename_vars({'density0': 'density'})

    ds['tke0'] = xr.ones_like(ds['wind_speed'])
    ds['tke0'].attrs['units'] = ds['tke'].units
    ds['tke0'].attrs['standard_name'] = '-'
    ds['tke0'].attrs['long_name'] = ds['tke'].long_name
    ds['tke0'][:,2,:,:] = ds['tke'][:,0,:,:].values # 150 m  
    ds = ds.drop_vars('tke')
    ds = ds.rename_vars({'tke0': 'tke'})

    #Save in csv format  
    df = create_dataframe(product=self.product,ds=ds, lon_near=lon_near, lat_near=lat_near, outfile=self.datafile, variable=self.variable[:-2], start_time = self.start_time, end_time = self.end_time, save_csv=save_csv,save_nc=save_nc, height=self.height)    
    ds.close()
   #remove temp/cache files
    if save_cache == False:
        for i in range(len(date_list)):
            try:
                os.remove(tempfile[i])
            except PermissionError:
                print(f"Skipping deletion of {tempfile[i]} due to PermissionError")
    
    return df

def NORA3_stormsurge_ts(self, save_csv = False,save_nc = False, save_cache =False):
    """
    Extract times series of  the nearest gird point (lon,lat) from
    nora3 sea level dataset and save it as netcdf.
    """
    #self.variable.append('lon_rho') # keep info of regular lon
    #self.variable.append('lat_rho')  # keep info of regular lat
    date_list = get_date_list(product=self.product, start_date=self.start_time, end_date=self.end_time)        
    
    tempfile = tempfile_dir(self.product,self.lon, self.lat, date_list,dirName="cache")

    # extract point and create temp files
    for i in range(len(date_list)):
        x_coor_str, y_coor_str, infile = get_url_info(product=self.product, date=date_list[i])
             
        if i==0:
            x_coor, y_coor, lon_near, lat_near = get_near_coord(infile=infile, lon=self.lon, lat=self.lat, product=self.product)        

        opt = ['-O -v '+",".join(self.variable)+' -d '+x_coor_str+','+str(x_coor[0])+' -d '+y_coor_str+','+str(y_coor[0])]

        apply_nco(infile,tempfile[i],opt)
    
    check_datafile_exists(self.datafile)
    
    #merge temp files
    ds = xr.open_mfdataset(paths=tempfile[:])
    ds = ds.rename_dims({'ocean_time': 'time'})   
    ds = ds.rename_vars({'ocean_time': 'time'})   
    #Save in csv format    
    df = create_dataframe(product=self.product,ds=ds, lon_near=lon_near, lat_near=lat_near, outfile=self.datafile, variable=self.variable, start_time = self.start_time, end_time = self.end_time, save_csv=save_csv,save_nc = save_nc, height=self.height)    
    ds.close()
   #remove temp/cache files
    if save_cache == False:
        for i in range(len(date_list)):
            try:
                os.remove(tempfile[i])
            except PermissionError:
                print(f"Skipping deletion of {tempfile[i]} due to PermissionError")
    
    return df

def NORKYST800_ts(self, save_csv = False, save_nc = False, save_cache = False):
    """
    Extract times series of  the nearest gird point (lon,lat) from
    Norkyst800.
    """
    self.variable.append('lon') # keep info of regular lon
    self.variable.append('lat')  # keep info of regular lat
    date_list = get_date_list(product=self.product, start_date=self.start_time, end_date=self.end_time)        

    tempfile = tempfile_dir(self.product,self.lon, self.lat, date_list,dirName="cache")

    # extract point and create temp files
    for i in range(len(date_list)):
        x_coor_str, y_coor_str, infile = get_url_info(product=self.product, date=date_list[i])

        if i==0 or date_list[i].strftime('%Y-%m-%d %H:%M:%S') == '2019-02-27 00:00:00': # '2019-02-27' change to new model set up
            x_coor, y_coor, lon_near, lat_near = get_near_coord(infile=infile, lon=self.lon, lat=self.lat, product=self.product)        

        opt = ['-O -v '+",".join(self.variable)+' -d '+x_coor_str+','+str(x_coor.values[0])+' -d '+y_coor_str+','+str(y_coor.values[0])]
        
        apply_nco(infile,tempfile[i],opt)
        remove_dimensions_from_netcdf(tempfile[i], dimensions_to_remove=['X', 'Y'])

    check_datafile_exists(self.datafile)
    #merge temp files
    existing_files = [f for f in tempfile if os.path.exists(f)]
    ds = xr.open_mfdataset(paths=existing_files[:])
    #Save in csv format   
    df = create_dataframe(product=self.product,ds=ds, lon_near=lon_near, lat_near=lat_near, outfile=self.datafile, variable=self.variable[:-2], start_time = self.start_time, end_time = self.end_time, save_csv=save_csv,save_nc = save_nc, height=self.height)    
    ds.close()
   #remove temp/cache files
    if save_cache == False:
        for i in range(len(date_list)):
            try:
                os.remove(tempfile[i])
            except PermissionError:
                print(f"Skipping deletion of {tempfile[i]} due to PermissionError")
    
    return df



def NORA3_combined_ts(self, save_csv = True,save_nc = False, save_cache =False):
    self.variable = ['hs','tp','fpI', 'tm1','tm2','tmp','Pdir','thq', 'hs_sea','tp_sea','thq_sea' ,'hs_swell','tp_swell','thq_swell']
    self.product = 'NORA3_wave_sub'
    df_wave = NORA3_wind_wave_ts(self, save_csv=True)
    top_header_wave = read_commented_lines(self.datafile)
    os.remove(self.datafile)
    self.variable = ['wind_speed','wind_direction']
    self.product = 'NORA3_wind_sub' 
    df_wind = NORA3_wind_wave_ts(self, save_csv=True)
    top_header_wind = read_commented_lines(self.datafile)
    os.remove(self.datafile)
    
    # merge dataframes
    df = df_wind.join(df_wave)
    top_header = np.append(top_header_wave,top_header_wind) 
    if save_csv == True:
        df.to_csv(self.datafile, index_label='time')
        with open(self.datafile, 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            for k in range(len(top_header)-1):                
                f.write(top_header[k].rstrip('\r\n') + '\n' )
            f.write(top_header[-1].rstrip('\r\n') + '\n' + content)

    print('Data saved at: ' +self.datafile)

    return df

def NorkystDA_surface_ts(self, save_csv = False,save_nc = False, save_cache =False):
    """
    Extract times series of  the nearest gird point (lon,lat) from
    norkystDA surface dataset and save it as netcdf or csv.
    """
    #self.variable.append('lon_rho') # keep info of regular lon
    #self.variable.append('lat_rho')  # keep info of regular lat
    date_list = get_date_list(product=self.product, start_date=self.start_time, end_date=self.end_time)        
    
    tempfile = tempfile_dir(self.product,self.lon, self.lat, date_list,dirName="cache")

    # extract point and create temp files
    for i in range(len(date_list)):
        x_coor_str, y_coor_str, infile = get_url_info(product=self.product, date=date_list[i])
             
        if i==0:
            x_coor, y_coor, lon_near, lat_near = get_near_coord(infile=infile, lon=self.lon, lat=self.lat, product=self.product)        

        opt = ['-O -v '+",".join(self.variable)+' -d '+x_coor_str+','+str(x_coor[0])+' -d '+y_coor_str+','+str(y_coor[0])]

        apply_nco(infile,tempfile[i],opt)
    
    check_datafile_exists(self.datafile)
    
    #merge temp files
    ds = xr.open_mfdataset(paths=tempfile[:])
    #Save in csv format    
    df = create_dataframe(product=self.product,ds=ds, lon_near=lon_near, lat_near=lat_near, outfile=self.datafile, variable=self.variable, start_time = self.start_time, end_time = self.end_time, save_csv=save_csv,save_nc = save_nc, height=self.height)    
    ds.close()
   #remove temp/cache files
    if save_cache == False:
        for i in range(len(date_list)):
            try:
                os.remove(tempfile[i])
            except PermissionError:
                print(f"Skipping deletion of {tempfile[i]} due to PermissionError")
    
    return df

def NorkystDA_zdepth_ts(self, save_csv = False,save_nc = False, save_cache =False):
    """
    Extract times series of  the nearest gird point (lon,lat) from
    norkystDA surface dataset and save it as netcdf or csv.
    """
    #self.variable.append('lon_rho') # keep info of regular lon
    #self.variable.append('lat_rho')  # keep info of regular lat
    date_list = get_date_list(product=self.product, start_date=self.start_time, end_date=self.end_time)        
    
    tempfile = tempfile_dir(self.product,self.lon, self.lat, date_list,dirName="cache")

    # extract point and create temp files
    for i in range(len(date_list)):
        x_coor_str, y_coor_str, infile = get_url_info(product=self.product, date=date_list[i])
             
        if i==0:
            x_coor, y_coor, lon_near, lat_near = get_near_coord(infile=infile, lon=self.lon, lat=self.lat, product=self.product)        

        opt = ['-O -v '+",".join(self.variable)+' -d '+x_coor_str+','+str(x_coor[0])+' -d '+y_coor_str+','+str(y_coor[0])]

        apply_nco(infile,tempfile[i],opt)
    
    check_datafile_exists(self.datafile)
    
    #merge temp files
    ds = xr.open_mfdataset(paths=tempfile[:])
    #Save in csv format    
    df = create_dataframe(product=self.product,ds=ds, lon_near=lon_near, lat_near=lat_near, outfile=self.datafile, variable=self.variable, start_time = self.start_time, end_time = self.end_time, save_csv=save_csv,save_nc = save_nc, height=self.height, depth = self.depth)    
    ds.close()
   #remove temp/cache files
    if save_cache == False:
        for i in range(len(date_list)):
            try:
                os.remove(tempfile[i])
            except PermissionError:
                print(f"Skipping deletion of {tempfile[i]} due to PermissionError")

    return df

def tempfile_dir(product,lon,lat, date_list,dirName):
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


def apply_nco(infile,tempfile,opt):
    nco = Nco()
    for x in range(0, 6):  # try 6 times
        try:
            nco.ncks(input=infile , output=tempfile, options=opt)
        except:
            print('......Retry'+str(x)+'.....')
            time.sleep(10)  # wait for 10 seconds before re-trying
        else:
            break

def OBS_E39(self, save_csv = False, save_nc = False, save_cache =False):
    """
    Extract times series of metocean E39 observations and save it as netcdf/csv.
    """
    self.variable.append('longitude') # keep info of  lon
    self.variable.append('latitude')  # keep info of  lat
    date_list = get_date_list(product=self.product, start_date=self.start_time, end_date=self.end_time) 
    tempfile = tempfile_dir(self.product,self.lon, self.lat, date_list,dirName="cache")

    # extract point and create temp files
    for i in range(len(date_list)):
        x_coor_str, y_coor_str, infile = get_url_info(product=self.product, date=date_list[i])
             
        opt = ['-O -v '+",".join(self.variable)]
        apply_nco(infile,tempfile[i],opt)
        
    
    check_datafile_exists(self.datafile)
    
    #merge temp files
    ds = xr.open_mfdataset(paths=tempfile[:])
    #Save in csv format    
    self.datafile.replace(self.datafile.split('_')[-3],'lat'+str(np.round(ds.latitude.mean().values,2)))
    self.datafile.replace(self.datafile.split('_')[-4],'lon'+str(np.round(ds.longitude.mean().values,2)))
    df = create_dataframe(product=self.product,ds=ds, lon_near=ds.longitude.mean().values, lat_near=ds.latitude.mean().values, outfile=self.datafile, variable=self.variable, start_time = self.start_time, end_time = self.end_time, save_csv=save_csv,save_nc = save_nc, height=self.height)    
    ds.close() 
   #remove temp/cache files
    if save_cache == False:
        for i in range(len(date_list)):
            try:
                os.remove(tempfile[i])
            except PermissionError:
                print(f"Skipping deletion of {tempfile[i]} due to PermissionError")

    print('Data saved at: ' +self.datafile)

    return df
