from metocean_api import ts


# Define TimeSeries-object
df_ts = ts.TimeSeries(lon=6.727, lat=65.064,start_time='2024-01-31', end_time='2024-02-01' , product='NORA3_wave')
#df_ts = ts.TimeSeries(lon=3.098, lat=52.48,start_time='2017-01-19', end_time='2017-02-20', product='ECHOWAVE')
#df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2021-01-14', end_time='2021-01-15' , product='NORA3_wind_sub')
#df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2021-01-01', end_time='2021-03-31' , product='NORA3_wave_sub')
#df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2000-01-01', end_time='2001-03-31' , product='NORA3_stormsurge')
#df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2021-01-01', end_time='2021-03-31' , product='NORA3_atm_sub')
#df_ts = ts.TimeSeries(lon=3.7, lat=61.8, start_time='2023-01-01', end_time='2023-02-01', product='NORA3_atm3hr_sub')
#df_ts = ts.TimeSeries(lon=3.73, lat=64.60,start_time='2020-09-14', end_time='2020-09-15', product='NORKYST800')
#df_ts = ts.TimeSeries(lon=3.73, lat=64.60,start_time='2017-01-19', end_time='2017-01-20' , product='NorkystDA_zdepth')
#df_ts = ts.TimeSeries(lon=3.73, lat=64.60,start_time='2017-01-19', end_time='2017-01-20' , product='NorkystDA_surface')


# Import data from thredds.met.no and save it as csv
df_ts.import_data(save_csv=True)
# Load data from a local csv-file
df_ts.load_data(local_file=df_ts.datafile)
