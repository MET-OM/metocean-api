from typing import Sequence
from . import read_metno as metno
from . import read_tudelft as tudelft
from . import read_ec as ec
from .aux_funcs import read_commented_lines


def combine_data(list_files, output_file=False):
    import pandas as pd
    import numpy as np

    for i in range(len(list_files)):
        df = pd.read_csv(list_files[i], comment="#", index_col=0, parse_dates=True)
        top_header = read_commented_lines(list_files[i])
        if i == 0:
            df_all = df
            top_header_all = top_header
        else:
            # df_all = df_all.join(df)
            df_all = pd.merge(
                df_all, df, how="outer", left_index=True, right_index=True
            )
            top_header_all = np.append(top_header_all, top_header)
    if output_file:
        df_all.to_csv(output_file, index_label="time")
        with open(output_file, "r+", encoding="utf8") as f:
            content = f.read()
            f.seek(0, 0)
            for k in range(len(top_header_all) - 1):
                f.write(top_header_all[k].rstrip("\r\n") + "\n")
            f.write(top_header_all[-1].rstrip("\r\n") + "\n" + content)
        print("Data saved at: " + output_file)
    return df_all


class TimeSeries:
    def __init__(
        self,
        lon: float,
        lat: float,
        start_time: str = "1990-01-01T00:00",
        end_time: str = "1991-12-31T23:59",
        variable: Sequence[str] = None,
        name: str = "AnonymousArea",
        product: str = "NORA3_wave_sub",
        datafile: str = None,
        data=None,
        height: Sequence[int] = None,
        depth: Sequence[float] = None,
    ):
        self.name = name
        self.lon = lon
        self.lat = lat
        self.product = product
        self.start_time = start_time
        self.end_time = end_time
        self.variable = variable
        self.height = height
        self.depth = depth
        if variable is None:
            # TODO: Handle None in later stages
            self.variable = []
        if height is None:
            # TODO: rather handle None in later stages
            # None should mean the user just wants all available levels
            self.height = [10, 20, 50, 100, 250, 500, 750]
        if depth is None:
            # TODO: rather handle None in later stages
            self.depth = [0, 1, 2.5, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500, 750, 1000, 1250,1500, 1750, 2000, 2250, 2500]
        if datafile is None:
            self.datafile = str(product+'_lon'+str(self.lon)+'_lat'+str(self.lat)+'_'+self.start_time.replace('-','')+'_'+self.end_time.replace('-','')+'.csv')
        self.data = data

    def import_data(self, save_csv=True, save_nc=False, use_cache=False):
        if (self.product == "NORA3_wave_sub") or (self.product == "NORA3_wave"):
            if self.variable == []:
                self.variable = [
                    "hs",
                    "tp",
                    "fpI",
                    "tm1",
                    "tm2",
                    "tmp",
                    "Pdir",
                    "thq",
                    "hs_sea",
                    "tp_sea",
                    "thq_sea",
                    "hs_swell",
                    "tp_swell",
                    "thq_swell",
                ]
            self.data = metno.nora3_wind_wave_ts(self, save_csv, save_nc, use_cache)
        elif self.product == "NORA3_wind_sub":
            if self.variable == []:
                self.variable = ["wind_speed", "wind_direction"]
            self.data = metno.nora3_wind_wave_ts(self, save_csv, save_nc, use_cache)
        elif self.product == "NORA3_wind_wave":
            self.data = metno.nora3_combined_ts(self, save_csv, save_nc, use_cache)
        elif self.product == "NORAC_wave":
            if self.variable == []:
                self.variable = [
                    "hs",
                    "tp",
                    "fpI",
                    "t0m1",
                    "t02",
                    "t01",
                    "dp",
                    "dir",
                    "phs0",
                    "ptp0",
                    "pdir0",
                    "phs1",
                    "ptp0",
                    "pdir1",
                ]
            else:
                pass
            self.data = metno.norac_ts(self, save_csv, save_nc, use_cache)
        elif self.product == "ERA5":
            if self.variable == []:
                self.variable = [
                    "100m_u_component_of_wind",
                    "100m_v_component_of_wind",
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                    "2m_temperature",
                    "instantaneous_10m_wind_gust",
                    "mean_direction_of_total_swell",
                    "mean_direction_of_wind_waves",
                    "mean_period_of_total_swell",
                    "mean_period_of_wind_waves",
                    "mean_wave_direction",
                    "mean_wave_period",
                    "peak_wave_period",
                    "significant_height_of_combined_wind_waves_and_swell",
                    "significant_height_of_total_swell",
                    "significant_height_of_wind_waves",
                ]
            else:
                pass
            self.data = ec.era5_ts(self, save_csv=save_csv, save_nc=save_nc)
        elif self.product == "GTSM":
            if self.variable == []:
                self.variable = [
                    "storm_surge_residual",
                    "tidal_elevation",
                    "total_water_level",
                ]
            else:
                pass
            self.data = ec.gtsm_ts(self, save_csv=save_csv, save_nc=save_nc)
        elif self.product == "NORA3_stormsurge":
            self.variable = ["zeta"]
            self.data = metno.nora3_stormsurge_ts(self, save_csv, save_nc, use_cache)
        elif self.product == "NORA3_atm_sub":
            if self.variable == []:
                self.variable = [
                    "air_pressure_at_sea_level",
                    "air_temperature_2m",
                    "relative_humidity_2m",
                    "surface_net_longwave_radiation",
                    "surface_net_shortwave_radiation",
                    "precipitation_amount_hourly",
                    "fog",
                ]
            self.data = metno.nora3_atm_ts(self, save_csv, save_nc, use_cache)
        elif self.product == "NORA3_atm3hr_sub":
            self.height = [50, 100, 150, 200, 300]
            self.variable = [
                "wind_speed",
                "wind_direction",
                "air_temperature",
                "relative_humidity",
                "density",
                "tke",
            ]
            self.data = metno.nora3_atm3hr_ts(self, save_csv, save_nc, use_cache)
        elif self.product == "NORKYST800":
            # FIXME: This means the user cannot decide, if we check for None later, we can rather depend on the data itself
            self.height = [
                0.0,
                3.0,
                10.0,
                15.0,
                25.0,
                50.0,
                75.0,
                100.0,
                150.0,
                200.0,
                250.0,
                300.0,
                500.0,
                1000.0,
                2000.0,
                3000.0,
            ]
            if self.variable == []:
                self.variable = ["salinity", "temperature", "u", "v", "zeta"]
            self.data = metno.norkyst_800_ts(self, save_csv, save_nc, use_cache)
        elif self.product.startswith("E39"):
            if self.variable == []:
                self.variable = ["Hm0"]
            self.data = metno.obs_e39(self, save_csv, save_nc, use_cache)
        elif self.product == "NorkystDA_surface":
            self.variable = ["u", "v", "zeta", "temp", "salt"]
            self.data = metno.norkyst_da_surface_ts(self, save_csv, save_nc, use_cache)
        elif self.product == "NorkystDA_zdepth":
            self.variable = ["u", "v", "zeta", "temp", "salt", "AKs"]
            self.data = metno.norkyst_da_zdepth_ts(self, save_csv, save_nc, use_cache)
        elif self.product == 'ECHOWAVE':
            self.variable = [ 'ucur', 'vcur', 'uwnd', 'vwnd', 'wlv', 'ice', 'hs', 'lm', 't02', 't01', 'fp', 'dir', 'spr', 'dp', 'phs0', 'phs1', 'phs2', 'ptp0', 'ptp1', 'ptp2', 'pdir0', 'pdir1']
            self.data = tudelft.echowave_ts(self, save_csv, save_nc, use_cache)

        return

    def load_data(self, local_file):
        import pandas as pd

        self.data = pd.read_csv(local_file, comment="#", index_col=0, parse_dates=True)
