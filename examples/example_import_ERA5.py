from metocean_api import ts

# Define TimeSeries-object
df_ts = ts.TimeSeries(lon=109.94, lat=15.51,start_time='2000-01-01', end_time='2000-12-31' , product='ERA5',
                      variable=[ 'significant_height_of_combined_wind_waves_and_swell',
                                'mean_wave_direction',
                               'peak_wave_period'])


#df_ts = ts.TimeSeries(lon=6, lat=55.7,start_time='2012-01-01', end_time='2012-01-31' , product='GTSM',
#                      variable=['storm_surge_residual','tidal_elevation','total_water_level'])

# list of wind and wave parameters in ERA5:
#        [    '100m_u_component_of_wind', '100m_v_component_of_wind', '10m_u_component_of_wind',
#            '10m_v_component_of_wind', '2m_temperature', 'instantaneous_10m_wind_gust',
#            'mean_direction_of_total_swell', 'mean_direction_of_wind_waves', 'mean_period_of_total_swell',
#            'mean_period_of_wind_waves', 'mean_wave_direction', 'mean_wave_period',
#            'peak_wave_period', 'significant_height_of_combined_wind_waves_and_swell', 'significant_height_of_total_swell',
#            'significant_height_of_wind_waves',
#        ]

# Import data from thredds.met.no and save it as csv
df_ts.import_data(save_csv=True,use_cache=True)

# Load data from a local csv-file
#df_ts.load_data(local_file=df_ts.datafile)
