from metocean_api import ts


# Define TimeSeries-object
df_ts = ts.TimeSeries(lon=3.73, lat=64.60,start_time='1990-01-01', end_time='2020-12-31' , product='NORA3_wind_wave')
#df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2021-01-01', end_time='2021-01-15' , product='NORA3_wind_sub')
#df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2021-01-01', end_time='2021-03-31' , product='NORA3_wave_sub')
#df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2000-01-01', end_time='2001-03-31' , product='NORA3_stormsurge')
#df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2021-01-01', end_time='2021-03-31' , product='NORA3_atm_sub')
#df_ts = ts.TimeSeries(lon=3.7, lat=61.8, start_time='2023-01-01', end_time='2023-02-01', product='NORA3_atm3hr_sub')


# Import data from thredds.met.no and save it as csv
df_ts.import_data(save_csv=True)

#print(df_ts.data)
# Load data from a local csv-file
#df_ts.load_data(local_file=df_ts.datafile)


