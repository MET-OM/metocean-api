from metocean_api import ts

def test_extract_NORA3wind():
    # Define TimeSeries-object
    df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2000-01-01', end_time='2000-01-31', product='NORA3_wind_sub')
    # Import data from thredds.met.no 
    df_ts.import_data(save_csv=False,save_nc=False)
    if df_ts.data.shape == (744,14):
        pass
    else:
        raise ValueError("Shape is not correct")    
    
def test_extract_NORA3wave():
    # Define TimeSeries-object
    df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2000-01-01', end_time='2000-01-31', product='NORA3_wave_sub')
    # Import data from thredds.met.no 
    df_ts.import_data(save_csv=False,save_nc=False)
    if df_ts.data.shape == (744,13):
        pass
    else:
        raise ValueError("Shape is not correct")    

#def test_extract_NORA3stormsurge():
#    # Define TimeSeries-object
#    df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2000-01-01', end_time='2000-01-31', product='NORA3_stormsurge')
#    # Import data from thredds.met.no 
#    df_ts.import_data(save_csv=False,save_nc=False)
#    if df_ts.data.shape == (744,1):
#        pass
#    else:
#        raise ValueError("Shape is not correct")    

def test_extract_NORA3atm():
    # Define TimeSeries-object
    df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2000-01-01', end_time='2000-01-31', product='NORA3_atm_sub')
    # Import data from thredds.met.no 
    df_ts.import_data(save_csv=False,save_nc=False)
    if df_ts.data.shape == (744,7):
        pass
    else:
        raise ValueError("Shape is not correct")    

def test_extract_NORA3atm3hr():
    # Define TimeSeries-object
    df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2000-01-01', end_time='2000-01-31', product='NORA3_atm3hr_sub')
    # Import data from thredds.met.no 
    df_ts.import_data(save_csv=False,save_nc=False)
    if df_ts.data.shape == (248,30):
        pass
    else:
        raise ValueError("Shape is not correct")    

def test_extract_OBS():
    # Define TimeSeries-object
    df_ts = ts.TimeSeries(lon='', lat='',start_time='2017-01-01', end_time='2017-01-31' , product='E39_B_Sulafjorden_wave', variable=['Hm0', 'tp'])
    # Import data from thredds.met.no 
    df_ts.import_data(save_csv=False,save_nc=False)
    if df_ts.data.shape == (4464,2):
        pass
    else:
        raise ValueError("Shape is not correct")    
