from metocean_api import ts

# Define TimeSeries-object
ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2022-10-01', end_time='2022-12-31' , product='NORA3_wind_wave')

# Import data from thredds.met.no and save it as csv
ts.import_ts(save_csv=True)

# Load data from a local csv-file
#ts.load_ts(local_file=ts.datafile)


