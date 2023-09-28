from metocean_api import ts

def test_extract_NORA3wind():
    # Define TimeSeries-object
    df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2000-01-01', end_time='2000-01-31', product='NORA3_wind_sub')
    # Import data from thredds.met.no and save it as csv
    df_ts.import_data(save_csv=False)

def test_extract_NORA3wave():
    # Define TimeSeries-object
    df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2000-01-01', end_time='2000-01-31', product='NORA3_wave_sub')
    # Import data from thredds.met.no and save it as csv
    df_ts.import_data(save_csv=False)

def test_extract_NORA3stormsurge():
    # Define TimeSeries-object
    df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2000-01-01', end_time='2000-01-31', product='NORA3_stormsurge')
    # Import data from thredds.met.no and save it as csv
    df_ts.import_data(save_csv=False)
