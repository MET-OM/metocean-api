from typing import List
import pandas as pd
import numpy as np
from .internal import metno,tudelft,ec
from .internal.aux_funcs import read_commented_lines

def combine_data(list_files, output_file=False):
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
        self.tempfiles: List[str] = []

    def __map_product_to_importer(self):
        for module in [metno, ec,tudelft]:
            importer = module.find_importer(self.product)
            if importer is not None:
                return importer
        raise ValueError(f"Product not recognized {self.product}")

    def import_data(self, save_csv=True, save_nc=False, use_cache=False):
        importer = self.__map_product_to_importer()
        self.data = importer(self, save_csv, save_nc, use_cache)

    def download(self):
        """
        Download the data to the cache and return the local file paths
        The cached files must be deleted manually when no longer needed
        
        Be aware that the data will be downloaded for the entire time range necessary to cover the start and end time, 
        but the time dimension will not be sliced. This must be done manually after loading the data.
        """
        importer = self.__map_product_to_importer()
        self.tempfiles =  importer(self, save_csv=False, save_nc=False, use_cache=True, download_only=True)
        return self.tempfiles

    def load_data(self, local_file):
        self.data = pd.read_csv(local_file, comment="#", index_col=0, parse_dates=True)
