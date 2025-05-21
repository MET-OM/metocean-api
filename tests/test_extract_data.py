import xarray as xr
import pandas as pd
from metocean_api import ts
from metocean_api.ts.internal import products
from metocean_api.ts.internal.convention import Convention

# Switches useful for local testing
USE_CACHE = True
SAVE_CSV = True
SAVE_NC = True

def test_extract_nora3_wind():
    df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2000-01-01', end_time='2000-01-31', product='NORA3_wind_sub')
    # Import data from thredds.met.no
    product = products.find_product(df_ts.product)
    assert product.name == df_ts.product
    assert product.convention == Convention.METEOROLOGICAL

    df_ts.import_data(save_csv=SAVE_CSV,save_nc=SAVE_NC, use_cache=USE_CACHE)
    assert (df_ts.lat_data, df_ts.lon_data) == (53.32374838481946, 1.3199893172215793)
    assert df_ts.data.shape == (744,14)
    __compare_loaded_data(df_ts)

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
    __compare_loaded_data(df_ts)

def test_nora3_wind_wave_combined():
    df_ts = ts.TimeSeries(lon=3.73, lat=64.60,start_time='2020-09-14', end_time='2020-09-15', product='NORA3_wind_wave', height=[10])
    df_ts.import_data(save_csv=SAVE_CSV,save_nc=SAVE_NC, use_cache=USE_CACHE)
    assert (df_ts.lat_data, df_ts.lon_data) == (64.60475157243123, 3.752025547482376)
    assert df_ts.data.shape == (48, 16)
    __compare_loaded_data(df_ts)

# def test_extract_nora3_stormsurge():
#    df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2000-01-01', end_time='2000-01-31', product='NORA3_stormsurge')
#    # Import data from thredds.met.no
#    df_ts.import_data(save_csv=SAVE_CSV,save_nc=SAVE_NC)
#    assert df_ts.data.shape == (744,1)


def __inferr_frequency(data: pd.DataFrame):
    inferred_freq = pd.infer_freq(data.index)
    # Set the inferred frequency if it’s detected
    if inferred_freq:
        data.index.freq = inferred_freq
    else:
        print("Could not infer frequency. Intervals may not be consistent.")

def __compare_loaded_data(df_ts: ts.TimeSeries):
    # Load the data back in and check that the data is the same
    df_ts2 = ts.TimeSeries(
        lon=df_ts.lon,
        lat=df_ts.lat,
        start_time=df_ts.start_time,
        end_time=df_ts.end_time,
        product=df_ts.product,
    )
    df_ts2.load_data(local_file=df_ts.datafile)
    __inferr_frequency(df_ts.data)
    __inferr_frequency(df_ts2.data)
    pd.testing.assert_frame_equal(df_ts.data, df_ts2.data)

def test_extract_nora3_fp():
    df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2000-01-01', end_time='2000-01-02', product='NORA3_fpc')
    # Import data from thredds.met.no
    df_ts.import_data(save_csv=SAVE_CSV,save_nc=SAVE_NC, use_cache=USE_CACHE)
    assert (df_ts.lat_data, df_ts.lon_data) == (53.32374838481946, 1.3199893172215793)
    assert df_ts.data.shape == (48, 206) 
    __compare_loaded_data(df_ts)


def test_extract_nora3_offshorewind():
    df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2000-01-01', end_time='2000-01-02', product='NORA3_offshore_wind')
    # Import data from thredds.met.no
    df_ts.import_data(save_csv=SAVE_CSV,save_nc=SAVE_NC, use_cache=USE_CACHE)
    assert (df_ts.lat_data, df_ts.lon_data) == (53.32374838481946, 1.3199893172215793)
    assert df_ts.data.shape == (48, 25) 
    __compare_loaded_data(df_ts)


def test_extract_nora3_():
    df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2000-01-01', end_time='2000-01-02', product='NORA3_')
    # Import data from thredds.met.no
    df_ts.import_data(save_csv=SAVE_CSV,save_nc=SAVE_NC, use_cache=USE_CACHE)
    assert (df_ts.lat_data, df_ts.lon_data) == (53.32374838481946, 1.3199893172215793)
    assert df_ts.data.shape == (9, 596)
    __compare_loaded_data(df_ts)

def test_extract_nora3_atm():
    df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2000-01-01', end_time='2000-01-31', product='NORA3_atm_sub')
    # Import data from thredds.met.no
    df_ts.import_data(save_csv=SAVE_CSV,save_nc=SAVE_NC, use_cache=USE_CACHE)
    assert (df_ts.lat_data, df_ts.lon_data) == (53.32374838481946, 1.3199893172215793)
    assert df_ts.data.shape == (744,7)
    __compare_loaded_data(df_ts)

def test_extract_nora3_atm3hr():
    df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2000-01-01', end_time='2000-01-31', product='NORA3_atm3hr_sub')
    # Import data from thredds.met.no
    df_ts.import_data(save_csv=SAVE_CSV,save_nc=SAVE_NC, use_cache=USE_CACHE)
    print(f"product: {df_ts.product}: {df_ts.lat_data}, {df_ts.lon_data}")
    assert (df_ts.lat_data, df_ts.lon_data) == (53.32374838481946, 1.3199893172215793)
    assert df_ts.data.shape == (248,30)
    __compare_loaded_data(df_ts)

def test_extract_obs():
    df_ts = ts.TimeSeries(lon='', lat='',start_time='2017-01-01', end_time='2017-01-31' , product='E39_B_Sulafjorden_wave', variable=['Hm0', 'tp'])
    # Import data from thredds.met.no
    df_ts.import_data(save_csv=SAVE_CSV,save_nc=SAVE_NC, use_cache=USE_CACHE)
    assert df_ts.data.shape == (4464,2)
    __compare_loaded_data(df_ts)

def test_norkyst_800():
    df_ts = ts.TimeSeries(lon=3.73, lat=64.60,start_time='2020-09-14', end_time='2020-09-15', product='NORKYST800')
    # Import data from thredds.met.no
    df_ts.import_data(save_csv=SAVE_CSV,save_nc=SAVE_NC, use_cache=USE_CACHE)
    assert (df_ts.lat_data, df_ts.lon_data) == (64.59832175874106, 3.728905373023728)
    assert df_ts.data.shape == (48, 65)
    __compare_loaded_data(df_ts)

def test_norkyst_da_zdepth():
    # We want to collect a subset
    depth = [0.0, 500.0, 2500.00]
    df_ts = ts.TimeSeries(lon=3.73, lat=64.60,start_time='2017-01-19', end_time='2017-01-20', product='NorkystDA_zdepth',depth=depth)
    # Import data from thredds.met.no
    df_ts.import_data(save_csv=SAVE_CSV,save_nc=SAVE_NC, use_cache=USE_CACHE)
    assert (df_ts.lat_data, df_ts.lon_data) == (64.59537563943964, 3.74450378868417)
    assert df_ts.data.shape == (24, 16)
    __compare_loaded_data(df_ts)

def test_norkyst_da_surface():
    df_ts = ts.TimeSeries(lon=3.73, lat=64.60,start_time='2017-01-19', end_time='2017-01-20', product='NorkystDA_surface')
    # Import data from thredds.met.no
    df_ts.import_data(save_csv=SAVE_CSV,save_nc=SAVE_NC, use_cache=USE_CACHE)
    assert (df_ts.lat_data, df_ts.lon_data) == (64.59537563943964, 3.74450378868417)
    assert df_ts.data.shape == (48, 5)
    __compare_loaded_data(df_ts)

def test_echowave():
    df_ts = ts.TimeSeries(lon=3.098, lat=52.48,start_time='2017-01-19', end_time='2017-01-20', product='ECHOWAVE')
    # Import data from https://data.4tu.nl/datasets/
    df_ts.import_data(save_csv=SAVE_CSV,save_nc=SAVE_NC, use_cache=USE_CACHE)
    assert df_ts.data.shape == (48, 22)
    __compare_loaded_data(df_ts)

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
    
def test_url_by_date():
    product = products.find_product("NORA3_atm_sub")
    urls = product.get_url_for_dates("2020-01-01","2020-12-31")
    assert len(urls) == 12
    for i in range(1,13):
        if i < 10:
            month = f"0{i}"
        else:
            month = i
        assert urls[i-1] == f"https://thredds.met.no/thredds/dodsC/nora3_subset_atmos/atm_hourly/arome3km_1hr_2020{month}.nc"
