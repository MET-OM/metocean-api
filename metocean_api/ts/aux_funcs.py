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
    y0 = dx.where(dx == dx.min(), drop=True).y
    x0 = dx.where(dx == dx.min(), drop=True).x
    return x0, y0

def get_url_spec(product, day):
    if product == 'SPEC_NORA3':
        url = 'https://thredds.met.no/thredds/dodsC/windsurfer/mywavewam3km_spectra/' + \
            day.strftime('%Y') + '/'+day.strftime('%m') + \
            '/SPC'+day.strftime('%Y%m%d')+'00.nc'
    elif product == 'SPEC_WW3':
        url = 'https://thredds.met.no/thredds/dodsC/ww3_4km_latest_files/' + \
            'ww3_POI_SPC_'+day.strftime('%Y%m%d')+'T00Z.nc'
    return url


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
    elif product == 'NORAC_wave':     
        infile = 'https://thredds.met.no/thredds/dodsC/norac_wave/field/ww3.'+date.strftime('%Y%m')+'.nc'
        x_coor_str = 'node'
        y_coor_str = 'node'
    print(infile)   
    return x_coor_str, y_coor_str, infile

def get_date_list(product, start_date, end_date):
    if product == 'NORA3_wave':
       date_list = pd.date_range(start=start_date , end=end_date, freq='D')
    elif product == 'NORA3_wave_sub':
        date_list = pd.date_range(start=start_date , end=end_date, freq='M')
    elif product == 'NORA3_wind_sub':
        date_list = pd.date_range(start=start_date , end=end_date, freq='M')
    elif product == 'NORAC_wave':
        date_list = pd.date_range(start=start_date , end=end_date, freq='M')
    return date_list

def drop_variables(product):
    if product == 'NORA3_wave':
       drop_var = ['projection_ob_tran','longitude','latitude']
    elif product == 'NORA3_wave_sub':
        drop_var = ['projection_ob_tran','longitude','latitude']
    elif product == 'NORA3_wind_sub':
        drop_var = ['projection_lambert','longitude','latitude','x','y','height']  
    elif product == 'NORAC_wave':
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
    elif product=='NORA3_wind_sub':
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
    print('Found nearest: lon.='+str(lon_near)+',lat.=' + str(lat_near))     
    return x_coor, y_coor, lon_near, lat_near

def create_dataframe(product,ds, lon_near, lat_near,outfile,variable, save_csv=True,  height=None):
    top_header = '#'+product + ' DATA. LONGITUDE:'+str(lon_near.round(4))+', LATITUDE:' + str(lat_near.round(4)) 
    if product=='NORA3_wind_sub': 
        ds0 = ds
        for i in range(len(height)):
            variable_height = [k + '_'+str(height[i])+'m' for k in variable]
            ds[variable_height] = ds0[variable].sel(height=height[i])
 
        ds = ds.drop_vars('height')
        ds = ds.drop([variable[0]])
        ds = ds.drop([variable[1]]) 
        ds = ds.drop('projection_lambert')
        ds = ds.drop('latitude')
        ds = ds.drop('longitude')
    else: 
        drop_var = drop_variables(product=product) 
        ds = ds.drop_vars(drop_var)

    df = ds.to_dataframe()
    df = df.astype(float).round(2)
    df.index = pd.DatetimeIndex(data=ds.time.values)

    #units = {'hs': 'm', 'tp': 's'}
    #units = []
    #for i in range(len(variable)):
    #    units.append(ds[variable[i]].units)
    
    #units = dict(zip(variable, units))
    #print(units)
    

    if save_csv == True:
        df.to_csv(outfile,index_label='time')
        with open(outfile, 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write(top_header.rstrip('\r\n') + '\n' + content)
    else:
        pass

    return df



