from __future__ import annotations
from typing import TYPE_CHECKING, override, Tuple, List
import os
from abc import abstractmethod
from tqdm import tqdm
import xarray as xr
from .. import aux_funcs
from ..product import Product


if TYPE_CHECKING:
    from ts.ts_mod import TimeSeries  # Only imported for type checking


class MetProduct(Product):
    """
    Base class for met.no products

    This class should be subclassed for each product.
    """

    @abstractmethod
    def get_default_variables(self):
        raise NotImplementedError(f"Not implemented for {self.name}")

    @abstractmethod
    def _get_url_info(self, date: str):
        raise NotImplementedError(f"Not implemented for {self.name}")

    @abstractmethod
    def _get_near_coord(self, url: str, lon: float, lat: float):
        raise NotImplementedError(f"Not implemented for {self.name}")

    @abstractmethod
    def get_dates(self, start_date, end_date):
        raise NotImplementedError(f"Not implemented for {self.name}")

    @override
    def import_data(self, ts: TimeSeries, save_csv=True, save_nc=False, use_cache=False):
        tempfiles, lon_near, lat_near = self.download_temporary_files(ts, use_cache)
        return self._combine_temporary_files(ts, save_csv, save_nc, use_cache, tempfiles, lon_near, lat_near, height=ts.height, depth=ts.depth)

    @override
    def download_temporary_files(self, ts: TimeSeries, use_cache: bool = False) -> Tuple[List[str], float, float]:
        if ts.variable == [] or ts.variable is None:
            ts.variable = self.get_default_variables()
        start_time = ts.start_time
        end_time = ts.end_time
        lat = ts.lat
        lon = ts.lon

        dates = self.get_dates(start_time, end_time)

        tempfiles = aux_funcs.get_tempfiles(self.name, lon, lat, dates)
        selection = None
        lon_near = None
        lat_near = None

        # extract point and create temp files
        for i in tqdm(range(len(dates))):
            url = self._get_url_info(dates[i])

            if i == 0:
                selection, lon_near, lat_near = self._get_near_coord(url, lon, lat)
                tqdm.write(f"Nearest point to lat.={lat:.3f},lon.={lon:.3f} was found at lat.={lat_near:.3f},lon.={lon_near:.3f}")

            if use_cache and os.path.exists(tempfiles[i]):
                tqdm.write(f"Found cached file {tempfiles[i]}. Using this instead")
            else:
                with xr.open_dataset(url) as dataset:
                    tqdm.write(f"Downloading {url}.")
                    # Reduce to the wanted variables and coordinates
                    dataset = dataset[ts.variable]
                    dataset = dataset.sel(selection).squeeze(drop=True)
                    dataset = self._alter_temporary_dataset_if_needed(dataset)
                    dataset.to_netcdf(tempfiles[i])

        ts.lat_data = lat_near
        ts.lon_data = lon_near
        return tempfiles, lon_near, lat_near

    def _alter_temporary_dataset_if_needed(self, dataset: xr.Dataset):
        # Override this method in subclasses to alter the dataset before saving it to a temporary file
        # Renaming variables that does not follow the standard names should also be done here
        return dataset

    def _combine_temporary_files(
        self, ts: TimeSeries, save_csv, save_nc, use_cache, tempfiles, lon_near, lat_near, **flatten_dims
    ):
        self._remove_if_datafile_exists(ts.datafile)
        # merge temp files
        with xr.open_mfdataset(tempfiles) as ds:
            if save_nc:
                # Save the unaltered structure
                ds = ds.sel({"time": slice(ts.start_time, ts.end_time)})
                ds.to_netcdf(ts.datafile.replace(".csv", ".nc"))

            df = self.create_dataframe(
                ds=ds,
                lon_near=lon_near,
                lat_near=lat_near,
                outfile=ts.datafile,
                start_time=ts.start_time,
                end_time=ts.end_time,
                save_csv=save_csv,
                **flatten_dims,
            )

        if not use_cache:
            # remove temp/cache files
            self._clean_cache(tempfiles)

        return df

    def create_dataframe(self, ds: xr.Dataset, lon_near, lat_near, outfile, start_time, end_time, save_csv=True, **flatten_dims):
        ds = self._flatten_data_structure(ds, **flatten_dims)
        return aux_funcs.create_dataframe(self.name, ds, lon_near, lat_near, outfile, start_time, end_time, save_csv)

    def _get_values_for_dimension(self, ds: xr.Dataset, flatten_dims, dim):
        if dim in ds.dims:
            levels = ds.variables[dim].values
            values = flatten_dims.get(dim)
            if values:
                for value in values:
                    if value not in levels:
                        flevels = [f"{x:.2f}" for x in levels]
                        raise ValueError(f"Value {value} not found in dimension {dim}. Available values are {flevels}")
                return values
            return levels
        return []

    def _flatten_data_structure(self, ds: xr.Dataset, **flatten_dims):
        return ds.drop_vars(self._drop_variables(), errors="ignore")

    def _drop_variables(self):
        return ["longitude", "latitude"]


    def _clean_cache(self, tempfiles):
        for tmpfile in tempfiles:
            try:
                os.remove(tmpfile)
            except PermissionError:
                print(f"Skipping deletion of {tmpfile} due to PermissionError")

    def _remove_if_datafile_exists(self, datafile):
        if os.path.exists(datafile):
            os.remove(datafile)
