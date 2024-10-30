import xarray as xr
from metocean_api import ts
from metocean_api.ts.internal import products

# Switches useful for local testing
USE_CACHE = False
SAVE_CSV = True
SAVE_NC = True

def test_extract_nora3_wind():
    df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2000-01-01', end_time='2000-01-31', product='NORA3_wind_sub')
    # Import data from thredds.met.no
    df_ts.import_data(save_csv=SAVE_CSV,save_nc=SAVE_NC, use_cache=USE_CACHE)
    assert (df_ts.lat_data, df_ts.lon_data) == (53.32374838481946, 1.3199893172215793)
    assert df_ts.data.shape == (744,14)

def test_download_of_temporary_files():
    # Pick a time region with a start and end time where the temporary files will cover more than the requested time
    df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2020-10-21', end_time='2020-11-21', product='NORA3_wave_sub')
    product = products.find_product(df_ts.product)
    files,lon_data,lat_data=product.download_temporary_files(df_ts)
    assert (lat_data, lon_data) == (53.32494354248047, 1.3358169794082642)
    assert len(files) == 2
    with xr.open_mfdataset(files) as values:
        # Make sure we have all the data in the temporary files
        hs = values["hs"]
        assert len(hs) == 1464
        # Slice the time series to the requested time
        hs = hs.sel(time=slice(df_ts.start_time, df_ts.end_time))
        assert len(hs) == 768

def test_extract_nora3_wave():
    df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2000-01-01', end_time='2000-01-31', product='NORA3_wave_sub')
    # Import data from thredds.met.no
    df_ts.import_data(save_csv=SAVE_CSV,save_nc=SAVE_NC, use_cache=USE_CACHE)
    assert (df_ts.lat_data, df_ts.lon_data) == (53.32494354248047, 1.3358169794082642)
    assert df_ts.data.shape == (744,14)

def test_nora3_wind_wave_combined():
    df_ts = ts.TimeSeries(lon=3.73, lat=64.60,start_time='2020-09-14', end_time='2020-09-15', product='NORA3_wind_wave', height=[10])
    # Import data from thredds.met.no
    df_ts.import_data(save_csv=SAVE_CSV,save_nc=SAVE_NC, use_cache=USE_CACHE)
    assert (df_ts.lat_data, df_ts.lon_data) == (64.60475157243123, 3.752025547482376)
    assert df_ts.data.shape == (48, 16)

#def test_extract_nora3_stormsurge():
#    df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2000-01-01', end_time='2000-01-31', product='NORA3_stormsurge')
#    # Import data from thredds.met.no
#    df_ts.import_data(save_csv=SAVE_CSV,save_nc=SAVE_NC)
#    assert df_ts.data.shape == (744,1)


def test_extract_nora3_atm():
    df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2000-01-01', end_time='2000-01-31', product='NORA3_atm_sub')
    # Import data from thredds.met.no
    df_ts.import_data(save_csv=SAVE_CSV,save_nc=SAVE_NC, use_cache=USE_CACHE)
    assert (df_ts.lat_data, df_ts.lon_data) == (53.32374838481946, 1.3199893172215793)
    assert df_ts.data.shape == (744,7)

def test_extract_nora3_atm3hr():
    df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2000-01-01', end_time='2000-01-31', product='NORA3_atm3hr_sub')
    # Import data from thredds.met.no
    df_ts.import_data(save_csv=SAVE_CSV,save_nc=SAVE_NC, use_cache=USE_CACHE)
    print(f"product: {df_ts.product}: {df_ts.lat_data}, {df_ts.lon_data}")
    assert (df_ts.lat_data, df_ts.lon_data) == (53.32374838481946, 1.3199893172215793)
    assert df_ts.data.shape == (248,30)

def test_extract_obs():
    df_ts = ts.TimeSeries(lon='', lat='',start_time='2017-01-01', end_time='2017-01-31' , product='E39_B_Sulafjorden_wave', variable=['Hm0', 'tp'])
    # Import data from thredds.met.no
    df_ts.import_data(save_csv=SAVE_CSV,save_nc=SAVE_NC, use_cache=USE_CACHE)
    assert df_ts.data.shape == (4464,2)

def test_norkyst_800():
    df_ts = ts.TimeSeries(lon=3.73, lat=64.60,start_time='2020-09-14', end_time='2020-09-15', product='NORKYST800')
    # Import data from thredds.met.no
    df_ts.import_data(save_csv=SAVE_CSV,save_nc=SAVE_NC, use_cache=USE_CACHE)
    assert (df_ts.lat_data, df_ts.lon_data) == (64.59832175874106, 3.728905373023728)
    assert df_ts.data.shape == (48, 65)

def test_norkyst_da_zdepth():
    # We want to collect a subset
    depth = [0.0, 500.0, 2500.00]
    df_ts = ts.TimeSeries(lon=3.73, lat=64.60,start_time='2017-01-19', end_time='2017-01-20', product='NorkystDA_zdepth',depth=depth)
    # Import data from thredds.met.no
    df_ts.import_data(save_csv=SAVE_CSV,save_nc=SAVE_NC, use_cache=USE_CACHE)
    assert (df_ts.lat_data, df_ts.lon_data) == (64.59537563943964, 3.74450378868417)
    assert df_ts.data.shape == (24, 16)

def test_norkyst_da_surface():
    df_ts = ts.TimeSeries(lon=3.73, lat=64.60,start_time='2017-01-19', end_time='2017-01-20', product='NorkystDA_surface')
    # Import data from thredds.met.no
    df_ts.import_data(save_csv=SAVE_CSV,save_nc=SAVE_NC, use_cache=USE_CACHE)
    assert (df_ts.lat_data, df_ts.lon_data) == (64.59537563943964, 3.74450378868417)
    assert df_ts.data.shape == (48, 5)

def test_echowave():
    df_ts = ts.TimeSeries(lon=3.098, lat=52.48,start_time='2017-01-19', end_time='2017-01-20', product='ECHOWAVE')
    # Import data from https://data.4tu.nl/datasets/
    df_ts.import_data(save_csv=SAVE_CSV,save_nc=SAVE_NC, use_cache=USE_CACHE)
    assert df_ts.data.shape == (48, 22)

def test_extract_nora3_wave_spectra():
    df_ts = ts.TimeSeries(lon=3.73, lat=64.60,start_time='2017-01-29',end_time='2017-02-02',product='NORA3_wave_spec')
    # Import data from thredds.met.no
    df_ts.import_data(save_csv=SAVE_CSV,save_nc=SAVE_NC, use_cache=USE_CACHE)
    assert (df_ts.lat_data, df_ts.lon_data) == (64.60214233398438, 3.7667124271392822)
    assert df_ts.data.shape == (120,30,24)

def test_extract_norac_wave_spectra():
    df_ts = ts.TimeSeries(lon=8, lat=64,start_time='2017-01-01',end_time='2017-01-04',product='NORAC_wave_spec')
    # Import data from thredds.met.no
    df_ts.import_data(save_csv=SAVE_CSV,save_nc=SAVE_NC, use_cache=USE_CACHE)
    assert (df_ts.lat_data, df_ts.lon_data) == (64.03120422363281, 7.936006546020508)
    assert df_ts.data.shape == (744,45,36)
