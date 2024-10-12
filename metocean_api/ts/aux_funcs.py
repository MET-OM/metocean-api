from __future__ import annotations
import os
import xarray as xr
import pandas as pd
import numpy as np
import cartopy.crs as ccrs

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

    if distance.min() > 50: # min distance between requested and available grid points above 50 km
        raise ValueError("Requested grid point out of model domain!!!")

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
    elif product == 'NORKYST800':     
        if date>=pd.Timestamp('2016-09-14 00:00:00') and date<=pd.Timestamp('2019-02-26 00:00:00'):
            infile = 'https://thredds.met.no/thredds/dodsC/sea/norkyst800mv0_1h/NorKyst-800m_ZDEPTHS_his.an.'+date.strftime('%Y%m%d%H')+'.nc'
        elif date>pd.Timestamp('2019-02-26 00:00:00'):
            infile = 'https://thredds.met.no/thredds/dodsC/fou-hi/norkyst800m-1h/NorKyst-800m_ZDEPTHS_his.an.'+date.strftime('%Y%m%d%H')+'.nc'
        x_coor_str = 'X'
        y_coor_str = 'Y'
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
    elif product == 'ECHOWAVE': 
        infile = 'https://opendap.4tu.nl/thredds/dodsC/data2/djht/f359cd0f-d135-416c-9118-e79dccba57b9/1/{}/TU-MREL_EU_ATL-2M_{}.nc'.format(date.strftime('%Y'),date.strftime('%Y%m')) 
        x_coor_str = 'longitude'
        y_coor_str = 'latitude'
    else:
        raise ValueError(f'Product not found {product}')
    print(infile)   
    return x_coor_str, y_coor_str, infile

def get_date_list(product, start_date, end_date):
    from datetime import datetime
    if product == 'NORA3_wave' or product == 'ERA5' or product.startswith('NorkystDA'):
       return pd.date_range(start=start_date , end=end_date, freq='D')
    elif product == 'NORA3_wave_sub':
        date_list = pd.date_range(start=datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y-%m') , end=datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y-%m'), freq='MS')
    elif product == 'ECHOWAVE':
        date_list = pd.date_range(start=datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y-%m') , end=datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y-%m'), freq='MS')
    elif product == 'NORA3_wind_sub' or product == 'NORA3_atm_sub' or product == 'NORA3_atm3hr_sub':
        date_list = pd.date_range(start=datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y-%m') , end=datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y-%m'), freq='MS')
    elif product == 'NORAC_wave':
        date_list = pd.date_range(start=datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y-%m') , end=datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y-%m'), freq='MS')
    elif product == 'NORA3_stormsurge':
        return pd.date_range(start=start_date , end=end_date, freq='YS')
    elif product == 'NORKYST800':
        return pd.date_range(start=start_date , end=end_date, freq='D')
    elif product.startswith('E39'):
        return pd.date_range(start=start_date , end=end_date, freq='MS')
    raise ValueError(f'Product not found {product}')

def drop_variables(product: str):
    if product == 'NORA3_wave':
       drop_var = ['projection_ob_tran','longitude','latitude']
    elif product == 'ECHOWAVE':
       drop_var = ['longitude','latitude']
    elif product == 'NORA3_wave_sub':
        return ['longitude','latitude','rlat','rlon']
    elif product == 'NORA3_wind_sub':
        return ['projection_lambert','longitude','latitude','x','y','height']
    elif product == 'NORA3_atm_sub':
        return ['projection_lambert','longitude','latitude','x','y']  
    elif product == 'NORA3_atm3hr_sub':
        return  ['projection_lambert','longitude','latitude','x','y','height'] 
    elif product == 'NORAC_wave':
        drop_var = ['longitude','latitude'] 
    elif product == 'ERA5':
        return  ['longitude','latitude']  
    elif product == 'GTSM':
        return  ['station_x_coordinate','station_y_coordinate', 'stations']  
    elif product == 'NORA3_stormsurge':
        return  ['lon_rho','lat_rho']  
    elif product == 'NORKYST800':
        return  ['lon','lat']  
    elif product.startswith('E39'):
        return  ['longitude','latitude']
    elif product.startswith('NorkystDA'):
        drop_var = ['lon','lat','x','y']
        if 'zdepth' in product:
            drop_var.append('depth')

        return drop_var
    return []

def get_near_coord(infile, lon, lat, product):
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
    elif product=='ECHOWAVE':
        ds_point = ds.sel(longitude=lon,latitude=lat,method='nearest')
        lon_near = ds_point.longitude.values
        lat_near = ds_point.latitude.values
        x_coor = lon_near
        y_coor = lat_near
    elif product=='NorkystDA_surface' or 'NorkystDA_zdepth':
        x, y = find_nearest_cartCoord(ds.lon, ds.lat, lon, lat)
        lon_near = ds.lon.sel(y=y, x=x).values[0][0]
        lat_near = ds.lat.sel(y=y, x=x).values[0][0]  
        x_coor = x.values
        y_coor = y.values    
    else:
        print('Product Not Found!')
    print('Found nearest: lon.='+str(lon_near)+',lat.=' + str(lat_near))     
    return x_coor, y_coor, lon_near, lat_near

def create_dataframe(product,ds, lon_near, lat_near,outfile,variable, start_time, end_time, save_csv=True,save_nc=True, height=None, depth = None):
    if product=='NORA3_wind_sub': 
        for i in range(len(height)):
            variable_height = [k + '_'+str(height[i])+'m' for k in variable]
            ds[variable_height] = ds[variable].sel(height=height[i])

        ds = ds.drop_vars('height')
        ds = ds.drop_vars([variable[0]])
        ds = ds.drop_vars([variable[1]]) 
        ds = ds.drop_vars(['x','y'])
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
        ds = ds.drop_vars('projection_lambert',errors="ignore")
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
            spd, direction = uv2spddir(u, v)
            ds[f'current_speed_{depth[i]}m'] = xr.DataArray(spd, dims=('time'), attrs = {"standard_name": 'sea_water_speed', "units": "meter seconds-1"})
            ds[f'current_direction_{depth[i]}m'] = xr.DataArray(direction, dims=('time'), attrs = {"standard_name" : "sea_water_velocity_from_direction", "units": "degrees"})
            ds = ds.drop_vars([f'u_{depth[i]}m', 'v_{}m'.format(depth[i])])

        drop_var = drop_variables(product=product) 
        ds = ds.drop_vars(drop_var,errors="ignore")
        ds = ds.drop_vars(variable,errors="ignore")

    elif product == 'NorkystDA_surface':
        crs = ccrs.Stereographic(central_latitude=90, central_longitude=70, true_scale_latitude=60,false_easting=0, false_northing=0)
        angle = proj_rotation_angle(crs, ds)
        u, v = rotate_vectors_tonorth(angle, ds['u'].values, ds['v'].values)
        spd, direction = uv2spddir(u, v)
        ds['current_speed'] = xr.DataArray(spd, dims=('time'), attrs = {"standard_name": 'sea_water_speed', "units": "meter seconds-1"})
        ds['current_direction'] = xr.DataArray(direction, dims=('time'), attrs = {"standard_name" : "sea_water_velocity_from_direction", "units": "degrees"})
        ds = ds.drop_vars(['u', 'v'])

        drop_var = drop_variables(product=product)
        ds = ds.drop_vars(drop_var,errors="ignore")
    elif product=='NORKYST800': 
        ds0 = ds
        if 'depth' in ds['zeta'].dims:
            ds['zeta'] = ds.zeta.sel(depth=0)

        var_list = []
        for var_name in variable:
        # Check if 'depth' is not in the dimensions of the variable
            if 'depth' in ds[var_name].dims:
            # Append variable name to the list
                var_list.append(var_name)

        for i in range(len(height)):
            variable_height = [k + '_'+str(height[i])+'m' for k in var_list]
            ds[variable_height] = ds0[var_list].sel(depth=height[i],method='nearest')
        ds = ds.drop_vars(var_list)
        ds = ds.drop_vars('depth')
        ds = ds.squeeze(drop=True)

    else:
        drop_var = drop_variables(product=product) 
        ds = ds.drop_vars(drop_var,errors="ignore")
    # Check if 'valid_time' (e.g., ERA5) exists in the dataset dimensions, if not, use 'time' (NORA3,...)
    time_dim = 'valid_time' if 'valid_time' in ds.dims else 'time'
    ds = ds.sel({time_dim: slice(start_time, end_time)})
    df = ds.to_dataframe()
    df = df.astype(float).round(2)
    df.index = pd.DatetimeIndex(data=ds[time_dim].values)

    top_header = '#'+product + ';LONGITUDE:'+str(lon_near.round(4))+';LATITUDE:' + str(lat_near.round(4)) 
    list_vars = [i for i in ds.data_vars]
    vars_info = ['#Variable_name;standard_name;long_name;units']

    for var in list_vars:
        try:
            standard_name = ds[var].standard_name
        except AttributeError:
            standard_name = '-'
        try:
            long_name = ds[var].long_name
        except AttributeError:
            long_name = '-'

        vars_info = np.append(vars_info,"#"+var+";"+standard_name+";"+long_name+";"+ds[var].units)

    if save_csv:
        df.to_csv(outfile,index_label='time')
        with open(outfile, 'r+',encoding="utf8") as f:
            content = f.read()
            f.seek(0, 0)
            f.write((top_header).rstrip('\r\n') + '\n')
            for k in range(len(vars_info)-1):                
                f.write(vars_info[k].rstrip('\r\n') + '\n' )
            f.write(vars_info[-1].rstrip('\r\n') + '\n' + content)

    if save_nc:
        ds.to_netcdf(outfile.replace('csv','nc'))

    return df

def check_datafile_exists(datafile):
    if os.path.exists(datafile):
        os.remove(datafile)
        print(datafile, 'already exists, so it will be deleted and create a new....')

def read_commented_lines(datafile):
    commented_lines = []
    with open(datafile,encoding="utf8") as f:
        for line in f:
            if line.startswith("#"):
                commented_lines = np.append(commented_lines,line)
    return commented_lines

def day_list(start_time, end_time):
    """Determins a Pandas data range of all the days in the time span"""
    t0 = pd.Timestamp(start_time).strftime('%Y-%m-%d')
    t1 = pd.Timestamp(end_time).strftime('%Y-%m-%d')
    days = pd.date_range(start=t0, end=t1, freq='D')
    return days

def int_list_of_years(start_time, end_time):
    year0 = min(pd.Series(day_list(start_time, end_time)).dt.year)
    year1 = max(pd.Series(day_list(start_time, end_time)).dt.year)
    return np.linspace(year0,year1,year1-year0+1).astype(int)

def int_list_of_months(start_time, end_time):
    if len(int_list_of_years(start_time, end_time))>1:
        raise ValueError('Only use this function for times within a single year!')
    month0 = min(pd.Series(day_list(start_time, end_time)).dt.month)
    month1 = max(pd.Series(day_list(start_time, end_time)).dt.month)
    return np.linspace(month0,month1,month1-month0+1).astype(int)
