from __future__ import annotations # For TYPE_CHECKING

#from metocean_api.ts.aux_funcs import distance_2points, find_nearest_rotCoord, find_nearest_cartCoord, get_date_list, get_url_info, get_near_coord, create_dataframe
from .read_metno import *

class TimeSeries:
  def __init__(self, lon: float, lat: float, start_time: str='1990-01-01T00:00', end_time: str='1991-12-31T23:59',  
  name: str='AnonymousArea', product: str='NORA3_wave_sub', datafile: str='EmptyFile', data = [], height: int = [10, 20, 50, 100, 250, 500, 750]):
    self.name = name
    self.lon = lon
    self.lat = lat
    self.product = product
    self.start_time = start_time
    self.end_time = end_time
    self.variable = []
    self.height = height
    self.datafile = product+'_lon'+str(self.lon)+'_lat'+str(self.lat)+'_'+self.start_time.replace('-','')+'_'+self.end_time.replace('-','')+'.csv'
    self.data = data
    return

  def import_data(self, save_csv = True):
    if ((self.product=='NORA3_wave_sub') or (self.product=='NORA3_wave')):
      self.variable =  ['hs','tp','tm1','tm2','tmp','Pdir','thq', 'hs_sea','tp_sea','thq_sea' ,'hs_swell','tp_swell','thq_swell']
      self.data = NORA3_ts(self, save_csv = save_csv)
    elif self.product == 'NORA3_wind_sub':
       self.variable =  ['wind_speed','wind_direction']
       self.data = NORA3_ts(self, save_csv = save_csv)
    elif self.product == 'NORA3_wind_wave':
      self.data = NORA3_combined_ts(self, save_csv = save_csv)
    elif self.product == 'NORAC_wave':
      self.variable =  ['hs','tp','t0m1','t02','t01','dp','dir', 'phs0','ptp0','pdir0' ,'phs1','ptp0','pdir1']
      self.data = NORA3_ts(self, save_csv = save_csv) 
    return


  def load_data(self, local_file):
    import pandas as pd
    self.data = pd.read_csv(local_file,comment='#',index_col=0, parse_dates=True)





 


