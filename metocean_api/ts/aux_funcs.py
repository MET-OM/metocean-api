from __future__ import annotations

import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np
import time
import os
from nco import Nco

def distance_2points(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c  # in km
    return distance


def find_nearest_rotCoord(lon_model, lat_model, lon0, lat0):
    #print('find nearest point...')
    dx = distance_2points(lat0, lon0, lat_model, lon_model)
    rlat0 = dx.where(dx == dx.min(), drop=True).rlat
    rlon0 = dx.where(dx == dx.min(), drop=True).rlon
    return rlon0, rlat0

def find_nearest_cartCoord(lon_model, lat_model, lon0, lat0):
    #print('find nearest point...')
    dx = distance_2points(lat0, lon0, lat_model, lon_model)
    x_name = 'x' if 'x' in dx.dims else 'X' if 'X' in dx.dims else None
    x0 = dx.where(dx == dx.min(), drop=True)[x_name]    
    y_name = 'y' if 'y' in dx.dims else 'Y' if 'Y' in dx.dims else None
    y0 = dx.where(dx == dx.min(), drop=True)[y_name]    
    return x0, y0

def find_nearest_rhoCoord(lon_model, lat_model, lon0, lat0):
    #print('find nearest point...')
    dx = distance_2points(lat0, lon0, lat_model, lon_model)
    eta_rho0, xi_rho0 = np.where(dx.values == dx.values.min())
    return eta_rho0, xi_rho0

def get_url_info(product, date):
    if product == 'NORA3_wave':
        infile = 'https://thredds.met.no/thredds/dodsC/windsurfer/mywavewam3km_files/'+date.strftime('%Y')+'/'+date.strftime('%m')+'/'+date.strftime('%Y%m%d')+'_MyWam3km_hindcast.nc'
        x_coor_str = 'rlon'
        y_coor_str = 'rlat'
    elif product == 'NORA3_wave_sub':     
        infile = 'https://thredds.met.no/thredds/dodsC/nora3_subset_wave/wave_tser/'+date.strftime('%Y%m')+'_NORA3wave_sub_time_unlimited.nc'
        #infile = '/lustre/storeB/project/fou/om/NORA3/wave/wave_subset/'+date_list.strftime('%Y%m')[i]+'_NORA3wave_sub.nc'
        x_coor_str = 'rlon'
        y_coor_str = 'rlat'
    elif product == 'NORA3_wind_sub':     
        infile = 'https://thredds.met.no/thredds/dodsC/nora3_subset_atmos/wind_hourly/arome3kmwind_1hr_'+date.strftime('%Y%m')+'.nc'
        x_coor_str = 'x'
        y_coor_str = 'y'
    elif product == 'NORA3_atm_sub':   
        infile = 'https://thredds.met.no/thredds/dodsC/nora3_subset_atmos/atm_hourly/arome3km_1hr_'+date.strftime('%Y%m')+'.nc'
        x_coor_str = 'x'
        y_coor_str = 'y'
    elif product == 'NORA3_atm3hr_sub':   
        infile = 'https://thredds.met.no/thredds/dodsC/nora3_subset_atmos/atm_3hourly/arome3km_3hr_'+date.strftime('%Y%m')+'.nc'
        #infile = '/lustre/storeB/project/fou/om/NORA3/equinor/atm_v3/arome3km_3hr_'+date.strftime('%Y%m')+'.nc'
        x_coor_str = 'x'
        y_coor_str = 'y'
    elif product == 'NORAC_wave':     
        infile = 'https://thredds.met.no/thredds/dodsC/norac_wave/field/ww3.'+date.strftime('%Y%m')+'.nc'
        x_coor_str = 'node'
        y_coor_str = 'node'
    elif product == 'NORA3_stormsurge':     
        infile = 'https://thredds.met.no/thredds/dodsC/stormrisk/zeta_nora3era5_N4_'+date.strftime('%Y')+'.nc'
        #infile = 'https://thredds.met.no/thredds/dodsC/nora3_subset_sealevel/sealevel/zeta_nora3era5_N4_'+date.strftime('%Y')+'.nc'
        x_coor_str = 'eta_rho'
        y_coor_str = 'xi_rho'
    elif product == 'NORKYST800':     
        infile = 'https://thredds.met.no/thredds/dodsC/fou-hi/norkyst800m-1h/NorKyst-800m_ZDEPTHS_his.an.'+date.strftime('%Y%m%d%H')+'.nc'
        #https://thredds.met.no/thredds/dodsC/fou-hi/norkyst800m-1h/NorKyst-800m_ZDEPTHS_his.an.2024062200.nc
        x_coor_str = 'X'
        y_coor_str = 'Y'
    elif product.startswith('E39'):     
        infile = 'https://thredds.met.no/thredds/dodsC/obs/buoy-svv-e39/'+date.strftime('%Y/%m/%Y%m')+'_'+product+'.nc'
        x_coor_str = 'longitude'
        y_coor_str = 'latitude'  
    print(infile)   
    return x_coor_str, y_coor_str, infile

def get_date_list(product, start_date, end_date):
    if product == 'NORA3_wave' or product == 'ERA5':
       date_list = pd.date_range(start=start_date , end=end_date, freq='D')
    elif product == 'NORA3_wave_sub':
        date_list = pd.date_range(start=start_date , end=end_date, freq='MS')
    elif product == 'NORA3_wind_sub' or product == 'NORA3_atm_sub' or product == 'NORA3_atm3hr_sub':
        date_list = pd.date_range(start=start_date , end=end_date, freq='MS')
    elif product == 'NORAC_wave':
        date_list = pd.date_range(start=start_date , end=end_date, freq='MS')
    elif product == 'NORA3_stormsurge':
        date_list = pd.date_range(start=start_date , end=end_date, freq='YS')
    elif product == 'NORKYST800':
        date_list = pd.date_range(start=start_date , end=end_date, freq='D')
    elif product.startswith('E39'):
        date_list = pd.date_range(start=start_date , end=end_date, freq='MS')
    return date_list

def drop_variables(product):
    if product == 'NORA3_wave':
       drop_var = ['projection_ob_tran','longitude','latitude']
    elif product == 'NORA3_wave_sub':
        drop_var = ['projection_ob_tran','longitude','latitude']
    elif product == 'NORA3_wind_sub':
        drop_var = ['projection_lambert','longitude','latitude','x','y','height']  
    elif product == 'NORA3_atm_sub':
        drop_var = ['projection_lambert','longitude','latitude','x','y']  
    elif product == 'NORA3_atm3hr_sub':
        drop_var = ['projection_lambert','longitude','latitude','x','y','height'] 
    elif product == 'NORAC_wave':
        drop_var = ['longitude','latitude'] 
    elif product == 'ERA5':
        drop_var = ['longitude','latitude']  
    elif product == 'NORA3_stormsurge':
        drop_var = ['lon_rho','lat_rho']  
    elif product == 'NORKYST800':
        drop_var = ['lon','lat']  
    elif product.startswith('E39'):
        drop_var = ['longitude','latitude']
    return drop_var

def get_near_coord(infile, lon, lat, product):
    ds = xr.open_dataset(infile)
    print('Find nearest point to lon.='+str(lon)+','+'lat.='+str(lat))
    if ((product=='NORA3_wave_sub') or (product=='NORA3_wave')) :
        rlon, rlat = find_nearest_rotCoord(ds.longitude, ds.latitude, lon, lat)
        lon_near = ds.longitude.sel(rlat=rlat, rlon=rlon).values[0][0]
        lat_near = ds.latitude.sel(rlat=rlat, rlon=rlon).values[0][0]
        x_coor = rlon
        y_coor = rlat
    elif product=='NORA3_wind_sub' or product == 'NORA3_atm_sub' or product == 'NORA3_atm3hr_sub':
        x, y = find_nearest_cartCoord(ds.longitude, ds.latitude, lon, lat)
        lon_near = ds.longitude.sel(y=y, x=x).values[0][0]
        lat_near = ds.latitude.sel(y=y, x=x).values[0][0]  
        x_coor = x
        y_coor = y  
    elif product=='NORAC_wave':
        node_id = distance_2points(ds.latitude.values,ds.longitude.values,lat,lon).argmin()
        lon_near = ds.longitude.sel(node=node_id).values
        lat_near = ds.latitude.sel(node=node_id).values  
        x_coor = node_id
        y_coor = node_id
    elif product=='NORA3_stormsurge':
        eta_rho, xi_rho = find_nearest_rhoCoord(ds.lon_rho, ds.lat_rho, lon, lat)
        lon_near = ds.lon_rho.sel(eta_rho=eta_rho, xi_rho=xi_rho).values[0][0]
        lat_near = ds.lat_rho.sel(eta_rho=eta_rho, xi_rho=xi_rho).values[0][0]
        x_coor = eta_rho
        y_coor = xi_rho
    elif product=='NORKYST800':
        x, y = find_nearest_cartCoord(ds.lon, ds.lat, lon, lat)
        lon_near = ds.lon.sel(Y=y, X=x).values[0][0]
        lat_near = ds.lat.sel(Y=y, X=x).values[0][0]  
        x_coor = x
        y_coor = y
    print('Found nearest: lon.='+str(lon_near)+',lat.=' + str(lat_near))     
    return x_coor, y_coor, lon_near, lat_near

def create_dataframe(product,ds, lon_near, lat_near,outfile,variable, start_time, end_time, save_csv=True,save_nc=True, height=None):
    if product=='NORA3_wind_sub': 
        ds0 = ds
        for i in range(len(height)):
            variable_height = [k + '_'+str(height[i])+'m' for k in variable]
            ds[variable_height] = ds0[variable].sel(height=height[i])
 
        ds = ds.drop_vars('height')
        ds = ds.drop_vars([variable[0]])
        ds = ds.drop_vars([variable[1]]) 
        ds = ds.drop_vars('projection_lambert')
        ds = ds.drop_vars('latitude')
        ds = ds.drop_vars('longitude')
    elif product=='NORA3_atm3hr_sub': 
        ds0 = ds
        for i in range(len(height)):
            variable_height = [k + '_'+str(height[i])+'m' for k in variable]
            ds[variable_height] = ds0[variable].sel(height=height[i])

        ds = ds.drop_vars('height')
        ds = ds.drop_vars('wind_speed')
        ds = ds.drop_vars('wind_direction') 
        ds = ds.drop_vars('air_temperature')
        ds = ds.drop_vars('relative_humidity') 
        ds = ds.drop_vars('tke') 
        ds = ds.drop_vars('density') 
        ds = ds.drop_vars('projection_lambert')
        ds = ds.drop_vars('latitude')
        ds = ds.drop_vars('longitude')
    elif product=='NORKYST800': 
        ds0 = ds
        for i in range(len(height)):
            variable_height = [k + '_'+str(height[i])+'m' for k in variable]
            ds[variable_height] = ds0[variable].sel(depth=height[i])
        ds = ds.drop_vars(variable)
        ds = ds.drop_vars('depth')
        ds = ds.drop_vars('lat')
        ds = ds.drop_vars('lon')
        ds = ds.drop_vars('X')
        ds = ds.drop_vars('Y')
        ds = ds.drop_vars('projection_stere')
        ds = ds.squeeze(drop=True)
    else: 
        drop_var = drop_variables(product=product) 
        ds = ds.drop_vars(drop_var)
 
    ds = ds.sel(time=slice(start_time,end_time)) 
    df = ds.to_dataframe()
    df = df.astype(float).round(2)
    df.index = pd.DatetimeIndex(data=ds.time.values)
    
    top_header = '#'+product + ';LONGITUDE:'+str(lon_near.round(4))+';LATITUDE:' + str(lat_near.round(4)) 
    list_vars = [i for i in ds.data_vars]
    vars_info = ['#Variable_name;standard_name;long_name;units']

    for vars in list_vars:
        try:
            standard_name = ds[vars].standard_name
        except AttributeError as e:
            standard_name = '-'
        try:
            long_name = ds[vars].long_name
        except AttributeError as e:
            long_name = '-'
            
        vars_info = np.append(vars_info,"#"+vars+";"+standard_name+";"+long_name+";"+ds[vars].units)

    #try:
    #    institution = '#Institution;'+ds.institution
    #except AttributeError as e:
    #    institution = '#Institution;-' 

    if save_csv == True:
        df.to_csv(outfile,index_label='time')
        with open(outfile, 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write((top_header).rstrip('\r\n') + '\n')
            for k in range(len(vars_info)-1):                
                f.write(vars_info[k].rstrip('\r\n') + '\n' )
            f.write(vars_info[-1].rstrip('\r\n') + '\n' + content)
    
    if save_nc == True:
        ds.to_netcdf(outfile.replace('csv','nc'))
    
    return df

def check_datafile_exists(datafile):
    if os.path.exists(datafile):
        os.remove(datafile)
        print(datafile, 'already exists, so it will be deleted and create a new....')
    else:
        pass# print("....")
    return


def read_commented_lines(datafile):
    commented_lines = []
    with open(datafile) as f:
        for line in f:
            if line.startswith("#"):
                commented_lines = np.append(commented_lines,line)
    return commented_lines



