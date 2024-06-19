from __future__ import annotations
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np
import time
import os
import cartopy.crs as ccrs
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

def proj_xy_from_lonlat( proj, lon,lat):
    try:
        print(len(lon))
    except:
        lon = np.array([lon])
        lat = np.array([lat])
    transform = proj.transform_points(ccrs.PlateCarree(), lon, lat)
    x = transform[..., 0]
    y = transform[..., 1]
    return x[0],y[0]

def proj_rotation_angle(proj, ds):
    x0,y0 = proj_xy_from_lonlat(proj, 0, 90)
    return np.rad2deg(np.arctan2(x0-ds.x.values, y0-ds.y.values))

def rotate_vectors_tonorth(angle,ui,vi):
    #Rotate vectors
    cosa = np.cos(angle)
    sina = np.sin(angle)

    uo = (ui * cosa) - (vi * sina)
    vo = (ui * sina) + (vi * cosa)
    return uo, vo

def uv2spddir(u,v):
    '''Function that will calculate speed and direction from u- and v-components'''
    iSpd = np.sqrt((u**2) + (v**2))
    iDir = np.arctan2(u,v) * (180 / np.pi)
    iDir = np.where((iDir < 0),iDir + 360,iDir)
    return iSpd, iDir

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
    elif product.startswith('E39'):     
        infile = 'https://thredds.met.no/thredds/dodsC/obs/buoy-svv-e39/'+date.strftime('%Y/%m/%Y%m')+'_'+product+'.nc'
        x_coor_str = 'longitude'
        y_coor_str = 'latitude' 
    elif product == 'NorkystDA_surface': 
        infile = 'https://thredds.met.no/thredds/dodsC/nora3_subset_ocean/surface/{}/ocean_surface_2_4km-{}.nc'.format(date.strftime('%Y/%m'), date.strftime('%Y%m%d')) 
        x_coor_str = 'x'
        y_coor_str = 'y'
    elif product == 'NorkystDA_zdepth': 
        infile = 'https://thredds.met.no/thredds/dodsC/nora3_subset_ocean/zdepth/{}/ocean_zdepth_2_4km-{}.nc'.format(date.strftime('%Y/%m'), date.strftime('%Y%m%d')) 
        x_coor_str = 'x'
        y_coor_str = 'y'
    print(infile)   
    return x_coor_str, y_coor_str, infile

def get_date_list(product, start_date, end_date):
    if product == 'NORA3_wave' or product == 'ERA5' or product.startswith('NorkystDA'):
       date_list = pd.date_range(start=start_date , end=end_date, freq='D')
    elif product == 'NORA3_wave_sub':
        date_list = pd.date_range(start=start_date , end=end_date, freq='MS')
    elif product == 'NORA3_wind_sub' or product == 'NORA3_atm_sub' or product == 'NORA3_atm3hr_sub':
        date_list = pd.date_range(start=start_date , end=end_date, freq='MS')
    elif product == 'NORAC_wave':
        date_list = pd.date_range(start=start_date , end=end_date, freq='MS')
    elif product == 'NORA3_stormsurge':
        date_list = pd.date_range(start=start_date , end=end_date, freq='YS')
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
    elif product.startswith('E39'):
        drop_var = ['longitude','latitude']
    elif product.startswith('NorkystDA'):
        drop_var = ['lon','lat', 'projection_stere']
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
    elif product=='NorkystDA_surface' or 'NorkystDA_zdepth':
        eta_rho, xi_rho = find_nearest_rhoCoord(ds.lon, ds.lat, lon, lat)
        print(eta_rho, xi_rho)
        lon_near = ds.lon.isel(y=eta_rho[0], x=xi_rho[0]).values
        lat_near = ds.lat.isel(y=eta_rho[0], x=xi_rho[0]).values
        print(lon_near, lat_near)
        x_coor = xi_rho
        y_coor = eta_rho
    print('Found nearest: lon.='+str(lon_near)+',lat.=' + str(lat_near))     
    return x_coor, y_coor, lon_near, lat_near

def create_dataframe(product,ds, lon_near, lat_near,outfile,variable, start_time, end_time, save_csv=True,save_nc=True, height=None, depth = None):
    print(depth)
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

    elif product == 'NorkystDA_zdepth':
        ds0 = ds
        crs = ccrs.Stereographic(central_latitude=90, central_longitude=70, true_scale_latitude=60,false_easting=0, false_northing=0)
        angle = proj_rotation_angle(crs, ds)

        for i in range(len(depth)):
            variable_height = [k + '_'+str(depth[i])+'m' for k in variable]
            ds[variable_height] = ds0[variable].sel(depth=depth[i])
            # zeta is surface elevation, no point in adding for each depth. 
            if np.abs(depth[i]) > 0 and 'zeta_{}m'.format(depth[i]) in ds.keys():
                ds = ds.drop_vars('zeta_{}m'.format(depth[i]))
            
            u, v = rotate_vectors_tonorth(angle, ds['u_{}m'.format(depth[i])].values, ds['v_{}m'.format(depth[i])].values)
            spd, dir = uv2spddir(u, v)
            ds['current_speed_{}m'.format(depth[i])] = xr.DataArray(spd, dims=('time', 'y', 'x'), attrs = {"standard_name": 'sea_water_speed', "units": "meter seconds-1"})
            ds['current_direction_{}m'.format(depth[i])] = xr.DataArray(dir, dims=('time', 'y', 'x'), attrs = {"standard_name" : "sea_water_velocity_from_direction", "units": "degrees"})
            ds = ds.drop_vars(['u_{}m'.format(depth[i]), 'v_{}m'.format(depth[i])])
                
        drop_var = drop_variables(product=product) 
        ds = ds.drop_vars(drop_var)
        ds = ds.drop_vars(variable)

    elif product == 'NorkystDA_surface':
        crs = ccrs.Stereographic(central_latitude=90, central_longitude=70, true_scale_latitude=60,false_easting=0, false_northing=0)
        angle = proj_rotation_angle(crs, ds)
        u, v = rotate_vectors_tonorth(angle, ds['u'].values, ds['v'].values)
        spd, dir = uv2spddir(u, v)
        ds['current_speed'] = xr.DataArray(spd, dims=('time', 'y', 'x'), attrs = {"standard_name": 'sea_water_speed', "units": "meter seconds-1"})
        ds['current_direction'] = xr.DataArray(dir, dims=('time', 'y', 'x'), attrs = {"standard_name" : "sea_water_velocity_from_direction", "units": "degrees"})
        
        ds = ds.drop_vars(['u', 'v'])
        drop_var = drop_variables(product=product) 
        ds = ds.drop_vars(drop_var)

    else: 
        drop_var = drop_variables(product=product) 
        ds = ds.drop_vars(drop_var)
 
    ds = ds.sel(time=slice(start_time,end_time)) 
    df = ds.to_dataframe()
    print(df)
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



