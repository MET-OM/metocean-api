from metocean_api import ts


# Define TimeSeries-object
# Full list of obs. products at https://thredds.met.no/thredds/catalog/obs/buoy-svv-e39/catalog.html, e.g, product ='E39_B_Sulafjorden_wave',  'E39_F_Vartdalsfjorden_wave', 
df_ts = ts.TimeSeries(lon='', lat='',start_time='2017-01-01', end_time='2017-01-31' , product='E39_B_Sulafjorden_wave', variable=['Hm0', 'tp','thmax'])


# Import data from thredds.met.no and save it as csv
df_ts.import_data(save_csv=True, save_nc=False)

# Load data from a local csv-file
#df_ts.load_data(local_file=df_ts.datafile)



