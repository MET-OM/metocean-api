"""Product definition for importing data from a source"""

from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, List
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from ts.ts_mod import TimeSeries  # Only imported for type checking


class Product(ABC):
    """Abstract base class for a product that can be imported"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def import_data(self, ts: TimeSeries, save_csv=True, save_nc=False, use_cache=False):
        """Import data specified by the TimeSeries object"""
        raise NotImplementedError(f"import_data method not implemented for product {self.name}")

    @abstractmethod
    def download_temporary_files(self, ts: TimeSeries, use_cache: bool = False) -> Tuple[List[str], float, float]:
        """
        Download the data to the cache and return the local file paths together with the nearest grid point
        The cached files must be deleted manually when no longer needed
        
        Be aware that the data will be downloaded for the entire time range necessary to cover the start and end time, 
        but the time dimension will not be sliced. This must be done manually after loading the data.
        """
        raise NotImplementedError(f"import_data method not implemented for product {self.name}")
