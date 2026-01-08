import os
from pathlib import Path
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import itertools
import time
import sys
import threading

print('Cartopy loaded')

def distance_2points(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c  # in km

    if distance.min() > 50: # min distance between requested and available grid points above 50 km
        raise ValueError("Requested grid point out of model domain!!!")

    return distance


def find_nearest_rot_coord(lon_model, lat_model, lon0, lat0):
    #print('find nearest point...')
    dx = distance_2points(lat0, lon0, lat_model, lon_model)
    rlat0 = dx.where(dx == dx.min(), drop=True).rlat
    rlon0 = dx.where(dx == dx.min(), drop=True).rlon
    return rlon0, rlat0

def find_nearest_cart_coord(lon_model, lat_model, lon0, lat0):
    #print('find nearest point...')
    dx = distance_2points(lat0, lon0, lat_model, lon_model)
    x_name = 'x' if 'x' in dx.dims else 'X' if 'X' in dx.dims else None
    x0 = dx.where(dx == dx.min(), drop=True)[x_name]
    y_name = 'y' if 'y' in dx.dims else 'Y' if 'Y' in dx.dims else None
    y0 = dx.where(dx == dx.min(), drop=True)[y_name]
    return x0, y0

def __proj_xy_from_lonlat( proj, lon: float,lat: float):
    transform = proj.transform_points(ccrs.PlateCarree(), np.array([lon]), np.array([lat]))
    x = transform[..., 0]
    y = transform[..., 1]
    return x[0],y[0]

def proj_rotation_angle(proj, ds):
    x0,y0 = __proj_xy_from_lonlat(proj, 0, 90)
    return np.rad2deg(np.arctan2(x0-ds.x.values, y0-ds.y.values))

def rotate_vectors_tonorth(angle,ui,vi):
    #Rotate vectors
    cosa = np.cos(angle)
    sina = np.sin(angle)

    uo = (ui * cosa) - (vi * sina)
    vo = (ui * sina) + (vi * cosa)
    return uo, vo

def uv2spddir(u,v):
    '''Function that will calculate speed and direction from u- and v-components'''
    i_spd = np.sqrt((u**2) + (v**2))
    i_dir = np.arctan2(u,v) * (180 / np.pi)
    i_dir = np.where((i_dir < 0),i_dir + 360,i_dir)
    return i_spd, i_dir

def find_nearest_rho_coord(lon_model, lat_model, lon0, lat0):
    # print('find nearest point...')
    dx = distance_2points(lat0, lon0, lat_model, lon_model)
    eta_rho0, xi_rho0 = np.where(dx.values == dx.values.min())
    return eta_rho0, xi_rho0

def create_dataframe(product, ds: xr.Dataset, lon_near, lat_near, outfile, start_time, end_time, save_csv=True):
    ds = ds.sel({"time": slice(start_time, end_time)}).squeeze(drop=True)
    if len(ds.dims) > 1:
        raise ValueError(f"The dataset has more than one dimension: {ds.dims}. Please flatten the dataset before creating a dataframe.")
    df = ds.to_dataframe()
    df = df.astype(float, errors='ignore').round(2)

    header_lines = ["#" + product + ";LONGITUDE:" + str(lon_near.round(4)) + ";LATITUDE:" + str(lat_near.round(4))]
    header_lines.append("#Variable_name;standard_name;long_name;units")
    for name,vardata in ds.data_vars.items():
        varattr = vardata.attrs
        standard_name =varattr.get("standard_name", "-")
        long_name = varattr.get("long_name", "-")
        units = varattr.get("units", "-")
        header_lines.append("#" + name + ";" + standard_name + ";" + long_name + ";" + units)

    # Add column names last
    header_lines.append("time," + ",".join(df.columns))

    header = "\n".join(header_lines) + "\n"

    if save_csv:
        with open(outfile, "w", encoding="utf8", newline="") as f:
            f.write(header)
            df.to_csv(f, header=False, encoding=f.encoding, index_label="time")
            print(f"CSV file created at {outfile}")

    return df

def save_to_netcdf(ds, outfile):
    remove_if_datafile_exists(outfile)
    ds.to_netcdf(outfile)
    print(f"NetCDF file created at {outfile}")


def remove_if_datafile_exists(datafile):
    if os.path.exists(datafile):
        try:
            os.remove(datafile)
        except OSError as e:
            print(f"Error removing file {datafile}: {e}")


def read_commented_lines(datafile):
    commented_lines = []
    with open(datafile,encoding="utf8") as f:
        for line in f:
            if line.startswith("#"):
                commented_lines = np.append(commented_lines,line)
    return commented_lines


def get_tempfiles(product, lon, lat, dates):
    tempfiles = []
    dir_name = "cache"
    try:
        # Create target Directory
        os.mkdir(dir_name)
        print("Directory ", dir_name, " Created ")
    except FileExistsError:
        #Ignore the error
        pass

    for date in dates:
        tempfiles.append(str(Path(dir_name+"/"+product+"_"+"lon"+str(lon)+"lat"+str(lat)+"_"+date.strftime('%Y%m%d%H%M')+".nc")))

    return tempfiles


class Spinner:
    def __enter__(self):
        self.stop_spinner = threading.Event()
        self.spinner_thread = threading.Thread(target=self._spin)
        self.spinner_thread.start()
        return self

    def _spin(self):
        spinner = itertools.cycle(['-', '/', '|', '\\'])
        while not self.stop_spinner.is_set():
            sys.stdout.write(next(spinner))
            sys.stdout.flush()
            sys.stdout.write('\b')
            time.sleep(0.1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_spinner.set()
        self.spinner_thread.join()


def format_seconds_to_dhms(seconds):
    days = seconds // (24 * 3600)
    seconds %= (24 * 3600)

    hours = seconds // 3600
    seconds %= 3600

    minutes = seconds // 60
    seconds %= 60

    return f"{days} days, {hours}hours, {minutes}min"