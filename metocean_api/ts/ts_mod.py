from typing import List
import pandas as pd
import numpy as np
from .internal import products
from .internal.aux_funcs import read_commented_lines
from typing import Optional, List


def combine_data(list_files: List[str], output_file: Optional[str] = None):
    for i in range(len(list_files)):
        df = pd.read_csv(list_files[i], comment="#", index_col=0, parse_dates=True)
        top_header = read_commented_lines(list_files[i])
        if i == 0:
            df_all = df
            top_header_all = top_header
        else:
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
        variable: List[str] = None,
        name: str = "AnonymousArea",
        product: str = "NORA3_wave_sub",
        datafile: str = None,
        data=None,
        height: List[float] = None,
        depth: List[float] = None,
    ):
        self.name = name
        # The requested location
        self.lon = lon
        self.lat = lat
        # The actual location of the data (will be computed on fetch)
        self.lon_data = None
        self.lat_data = None
        self.product = product
        self.start_time = start_time
        self.end_time = end_time
        self.datafile = datafile
        self.variable = []
        self.height = height
        self.depth = depth
        if variable is not None:
            self.variable.extend(variable)
        if datafile is None:
            self.datafile = str(product+'_lon'+str(self.lon)+'_lat'+str(self.lat)+'_'+self.start_time.replace('-','')+'_'+self.end_time.replace('-','')+'.csv')
        self.data = data

    def import_data(self, save_csv=True, save_nc=False, use_cache=False):
        product = products.find_product(self.product)
        self.data = product.import_data(self, save_csv, save_nc, use_cache)

    def load_data(self, local_file):
        self.data = pd.read_csv(local_file, comment="#", index_col=0, parse_dates=True)
