from metocean_api import ts

# Define TimeSeries-object
df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2021-01-01', end_time='2021-03-31' , product='NORA3_wind_wave')
#df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2021-01-01', end_time='2022-03-31' , product='NORA3_stormsurge')


# Import data from thredds.met.no and save it as csv
df_ts.import_data(save_csv=True)

print(df_ts.data)

# Load data from a local csv-file
#df_ts.load_data(local_file=ts.datafile)

