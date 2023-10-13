from __future__ import annotations # For TYPE_CHECKING

from .read_metno import *
from .read_ec import *

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
      self.data = NORA3_wind_wave_ts(self, save_csv = save_csv)
    elif self.product == 'NORA3_wind_sub':
       self.variable =  ['wind_speed','wind_direction']
       self.data = NORA3_wind_wave_ts(self, save_csv = save_csv)
    elif self.product == 'NORA3_wind_wave':
      self.data = NORA3_combined_ts(self, save_csv = save_csv)
    elif self.product == 'NORAC_wave':
      self.variable =  ['hs','tp','t0m1','t02','t01','dp','dir', 'phs0','ptp0','pdir0' ,'phs1','ptp0','pdir1']
      self.data = NORAC_ts(self, save_csv = save_csv) 
    elif self.product == 'ERA5':
      self.variable = [
             'significant_height_of_combined_wind_waves_and_swell', 
        ]
#      self.variable = [
#            '100m_u_component_of_wind', '100m_v_component_of_wind', '10m_u_component_of_wind',
#            '10m_v_component_of_wind', '2m_temperature', 'instantaneous_10m_wind_gust',
#            'mean_direction_of_total_swell', 'mean_direction_of_wind_waves', 'mean_period_of_total_swell',
#            'mean_period_of_wind_waves', 'mean_wave_direction', 'mean_wave_period',
#            'peak_wave_period', 'significant_height_of_combined_wind_waves_and_swell', 'significant_height_of_total_swell',
#            'significant_height_of_wind_waves',
#        ] 
      self.data = ERA5_ts(self, save_csv = save_csv) 
    elif self.product == 'NORA3_stormsurge':
      self.variable =  ['zeta']
      self.data = NORA3_stormsurge_ts(self, save_csv = save_csv) 
    elif self.product == 'NORA3_atm_sub':
      self.variable =  ['air_pressure_at_sea_level', 'air_temperature_2m', 'relative_humidity_2m', 
                        'surface_net_longwave_radiation', 'surface_net_shortwave_radiation',
                        'precipitation_amount_hourly','fog']
      self.data = NORA3_atm_ts(self, save_csv = save_csv) 
    return

  def load_data(self, local_file):
    import pandas as pd
    self.data = pd.read_csv(local_file,comment='#',index_col=0, parse_dates=True)





 


