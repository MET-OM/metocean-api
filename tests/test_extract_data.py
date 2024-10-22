from metocean_api import ts

def test_extract_nora3_wind():
    df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2000-01-01', end_time='2000-01-31', product='NORA3_wind_sub')
    # Import data from thredds.met.no
    df_ts.import_data(save_csv=False,save_nc=False)
    assert df_ts.data.shape == (744,14)

def test_extract_nora3_wave():
    df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2000-01-01', end_time='2000-01-31', product='NORA3_wave_sub')
    # Import data from thredds.met.no
    df_ts.import_data(save_csv=False,save_nc=False)
    assert df_ts.data.shape == (744,14)

#def test_extract_nora3_stormsurge():
#    df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2000-01-01', end_time='2000-01-31', product='NORA3_stormsurge')
#    # Import data from thredds.met.no
#    df_ts.import_data(save_csv=False,save_nc=False)
#    assert df_ts.data.shape == (744,1)


def test_extract_nora3_atm():
    df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2000-01-01', end_time='2000-01-31', product='NORA3_atm_sub')
    # Import data from thredds.met.no
    df_ts.import_data(save_csv=False,save_nc=False)
    assert df_ts.data.shape == (744,7)

def test_extract_nora3_atm3hr():
    df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2000-01-01', end_time='2000-01-31', product='NORA3_atm3hr_sub')
    # Import data from thredds.met.no
    df_ts.import_data(save_csv=False,save_nc=False)
    assert df_ts.data.shape == (248,30)

def test_extract_obs():
    df_ts = ts.TimeSeries(lon='', lat='',start_time='2017-01-01', end_time='2017-01-31' , product='E39_B_Sulafjorden_wave', variable=['Hm0', 'tp'])
    # Import data from thredds.met.no
    df_ts.import_data(save_csv=False,save_nc=False)
    assert df_ts.data.shape == (4464,2)

def test_norkyst_800():
    df_ts = ts.TimeSeries(lon=3.73, lat=64.60,start_time='2020-09-14', end_time='2020-09-15', product='NORKYST800')
    # Import data from thredds.met.no
    df_ts.import_data(save_csv=False,save_nc=False)
    assert df_ts.data.shape == (48, 65)

def test_norkyst_da_zdepth():
    df_ts = ts.TimeSeries(lon=3.73, lat=64.60,start_time='2017-01-19', end_time='2017-01-20', product='NorkystDA_zdepth')
    # Import data from thredds.met.no
    df_ts.import_data(save_csv=False,save_nc=False)
    assert df_ts.data.shape == (24, 146)

def test_norkyst_da_surface():
    df_ts = ts.TimeSeries(lon=3.73, lat=64.60,start_time='2017-01-19', end_time='2017-01-20', product='NorkystDA_surface')
    # Import data from thredds.met.no
    df_ts.import_data(save_csv=False,save_nc=False)
    assert df_ts.data.shape == (48, 5)

def test_echowave():
    df_ts = ts.TimeSeries(lon=3.098, lat=52.48,start_time='2017-01-19', end_time='2017-01-20', product='ECHOWAVE')
    # Import data from https://data.4tu.nl/datasets/
    df_ts.import_data(save_csv=False,save_nc=False)
    assert df_ts.data.shape == (48, 22)

def test_extract_nora3_wave_spectra():
    df_ts = ts.TimeSeries(lon=3.73, lat=64.60,start_time='2017-01-29',end_time='2017-02-02',product='NORA3_wave_spec')
    # Import data from thredds.met.no
    df_ts.import_data(save_csv=False,save_nc=False)
    assert df_ts.data.shape == (120,30,24)
    
def test_extract_norac_wave_spectra():
    df_ts = ts.TimeSeries(lon=8, lat=64,start_time='2017-01-01',end_time='2017-01-04',product='NORAC_wave_spec')
    # Import data from thredds.met.no
    df_ts.import_data(save_csv=False,save_nc=False)
    assert df_ts.data.shape == (744,45,36)