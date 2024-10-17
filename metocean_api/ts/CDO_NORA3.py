
def cdo_NORA3(rlonmin,rlonmax,rlatmin,rlatmax,time_start,time_end,variables=False):
    #Download NORA3 Wave data from domain using polar coordinates rlon and rlat.
    #Variables must be a list, it is possible to choose from these:
    #NORA3_wave;
    #Variable_name;standard_name;long_name;units
    #Pdir;sea_surface_wave_to_direction_at_variance_spectral_density_maximu;peak direction;degree
    #fpI;interpolated_peak_frequency;interpolated peak frequency;s
    #hs;sea_surface_wave_significant_height;Total significant wave height;m
    #hs_sea;sea_surface_wind_wave_significant_height;Sea significant wave height;m
    #hs_swell;sea_surface_swell_wave_significant_height;Swell significant wave height;m
    #thq;sea_surface_wave_to_direction;Total mean wave direction;degree
    #thq_sea;sea_surface_wind_wave_to_direction;Sea mean wave direction;degree
    #thq_swell;sea_surface_swell_wave_to_direction;Swell mean wave direction;degree
    #tm1;sea_surface_wave_mean_period_from_variance_spectral_density_first_frequency_moment;Total m1-period;s
    #tm2;sea_surface_wave_mean_period_from_variance_spectral_density_second_frequency_moment;Total m2-period;s
    #tmp;sea_surface_wave_mean_period_from_variance_spectral_density_inverse_frequency_moment;Total mean period;s
    #tp;sea_surface_wave_period_at_variance_spectral_density_maximum;Total peak period;s
    #tp_sea;sea_surface_wind_wave_peak_period_from_variance_spectral_density;Sea peak period;s
    #tp_swell;sea_surface_swell_wave_peak_period_from_variance_spectral_density;Swell peak period;s
    import subprocess
    import pandas as pd
    lon_min, lon_max, lat_min, lat_max = rlonmin, rlonmax, rlatmin, rlatmax
    files = []
    for time in pd.date_range(time_start,time_end, freq="D"):        
        file = 'https://thredds.met.no/thredds/dodsC/windsurfer/mywavewam3km_files/'+time.strftime('%Y')+'/'+time.strftime('%m')+'/'+time.strftime('%Y%m%d')+'_MyWam3km_hindcast.nc'
        if variables == False:
            command = f"cdo sellonlatbox,{lon_min},{lon_max},{lat_min},{lat_max} {file} NORA3_{time.strftime('%Y%m%d')}.nc"
        else:
            command = f"cdo sellonlatbox,{lon_min},{lon_max},{lat_min},{lat_max} -selname,{','.join(variables)} {file} NORA3_{time.strftime('%Y%m%d')}.nc"
    
        subprocess.run(command, shell=True, check=True)
        print(f"NORA3_{time.strftime('%Y%m%d')}.nc added to your directory.")
        files.append(f'NORA3_{time.strftime("%Y%m%d")}.nc')
    #Merge the files created
    output = f'NORA3_{time_start}-{time_end}.nc'
    subprocess.run(f"cdo mergetime {' '.join(files)} {output}", shell=True, check=True) #wish to have file1.nc file2.nc etc
    return f"New file {output} is added to directory."

"""
####EXAMPLE:
variables = ['time','longitude','latitude', 'pDir', 'fpI','hs','hs_swell','thq','tm2','tmp','tp']  # replace with your actual variables
cdo_NORA3(8,16,15,24,"2021-02-01","2021-03-31",variables=variables)
"""