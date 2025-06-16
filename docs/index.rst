.. metocean-api documentation master file, created by
   sphinx-quickstart on Thu Sep 14 10:18:36 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to metocean-api's documentation!
========================================

**metocean-api**  is a Python tool designed to extract time series of metocean (meteorological and oceanographic) data from various sources, including global, regional, and coastal hindcasts and reanalysis.
The extracted data can be saved in CSV format or NetCDF for further analysis and usage.
Please look at the section below for more information about the available datasets and variables. 

Installing **metocean-api**
=============================================
Quick installation
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
.. code-block:: bash

   $ pip install metocean-api

or 

.. code-block:: bash

   $ conda install -c conda-forge metocean-api

Alternative 1: Using Mambaforge (alternative to Miniconda)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

1. Install `mambaforge <https://mamba.readthedocs.io/en/latest/installation.html>`_ (`download <https://github.com/conda-forge/miniforge#mambaforge>`_)
2. Set up a *Python 3* environment for metocean-api and install metocean-api

.. code-block:: bash

   $ mamba create -n metocean-api python=3 metocean-api
   $ conda activate metocean-api

Alternative 2: Using Mambaforge (alternative to Miniconda) and Git
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
1. Install `mambaforge <https://mamba.readthedocs.io/en/latest/installation.html>`_ (`download <https://github.com/conda-forge/miniforge#mambaforge>`_)
2. Clone metocean-api:

.. code-block:: bash

   $ git clone https://github.com/MET-OM/metocean-api.git
   $ cd metocean-api/

3. Create environment with the required dependencies and install metocean-api

.. code-block:: bash

  $ mamba env create -f environment.yml
  $ conda activate metocean-api
  $ pip install --no-deps -e .

This installs the metocean-api as an editable package. Therefore, you can directly change the repository or fetch the newest changes with :code:`git pull`. 

To update the local conda environment in case of new dependencies added to environment.yml:

.. code-block:: bash

  $ mamba env update -f environment.yml

Creating a TimeSeries-object
=====================================
This section documents the **ts-module**. The ts-object is initialized with the following python command:

.. code-block:: python

   from metocean_api import ts
   df_ts = ts.TimeSeries(lon=1.320, lat=53.324,
                      start_time='2000-01-01', end_time='2000-03-31' , 
                      product='NORA3_wind_wave') 


Available Datasets in metocean-api
=====================================

Several options for **product** are available. Please check the data catalog for the time coverage:

:warning: Parallel downloading is not allowed for NORA3 data. It's a breach of the terms of service for thredds.met.no and may result in the blocking of your computer.

:information: If you are using the default variables, using `use_cache=True` is recommended to reduce the load on the server and allow you to restart the download in case of failure.

* For wind NORA3 hourly data in 10, 20, 50, 100, 250, 500, 750m (Nordic Area) developed by MET Norway:

  .. code-block:: python
    
    product='NORA3_wind_sub'

  Dataset: https://thredds.met.no/thredds/catalog/nora3_subset_atmos/wind_hourly/catalog.html

* For atmospheric (pressure,temperature,precipitation,humidity, radiation) NORA3 hourly surface data (Nordic Area) developed by MET Norway:

  .. code-block:: python

    product='NORA3_atm_sub'

  Dataset: https://thredds.met.no/thredds/catalog/nora3_subset_atmos/atm_hourly/catalog.html

* For SST and atmospheric (wind, temperature, relative humidity, tke, air density) NORA3 3-hourly data in 50, 100, 150, 200, 300m (Nordic Area) developed by MET Norway: 

  .. code-block:: python
    
    product='NORA3_atm3hr_sub'

  Dataset: https://thredds.met.no/thredds/catalog/nora3_subset_atmos/atm_3hourly/catalog.html

* For wave NORA3 sub data (Nordic Seas) developed by MET Norway: 

  .. code-block:: python

    product='NORA3_wave_sub' 

  Dataset: https://thredds.met.no/thredds/catalog/nora3_subset_wave/wave_tser/catalog.html

* For combined wind and wave NORA3 sub data (Nordic Seas) developed by MET Norway: 

  .. code-block:: python

    product='NORA3_wind_wave'

* For wave NORA3 data (Nordic Seas + Arctic) developed by MET Norway: 

  .. code-block:: python
    
    product='NORA3_wave'

  Dataset: https://thredds.met.no/thredds/catalog/windsurfer/mywavewam3km_files/catalog.html

* For sea level NORA3 data (Nordic Seas) developed by MET Norway: 

  .. code-block:: python

    product='NORA3_stormsurge'

  Dataset: https://thredds.met.no/thredds/catalog/stormrisk/catalog.html

* For wave spectra from NORA3 (Nordic Seas) developed by MET Norway: 

  .. code-block:: python

    product='NORA3_wave_spec'

  Dataset: https://thredds.met.no/thredds/catalog/windsurfer/mywavewam3km_spectra/catalog.html

* For hourly raw output from NORA3 atmospheric hindcast developed by MET Norway: 

  .. code-block:: python

    product='NORA3_fpc'

  Dataset: https://thredds.met.no/thredds/projects/nora3.html (fc<YYYYMMDDHH>_<leadtime>_fp.nc files using [3-8] time lead to avoid time overlap)
  As the fluxes are accumulated overtime, we are taking the discrete time derivative from two consecutive forecasts ([acc_Hs(..._004_fp.nc) - acc_Hs(..._003_fp.nc)] / 3600s) 

* For 3hourly raw output product from NORA3 atmospheric hindcast developed by MET Norway: 

  .. code-block:: python

    product='NORA3_'

  Dataset: https://thredds.met.no/thredds/projects/nora3.html (fc<YYYYMMDDHH>_<leadtime>.nc files using [3,6] time lead to avoid time overlap)

* For wind data and Obukhov length over the NORA3 domain. This product is computed from the hourly raw data from the NORA3 atmospheric hindcast developed by MET Norway and a subset of atmospheric parameters. The available variables are wind speed and direction at 10, 20, 50, 100, 250, 500, and 750 meters with an hourly time step. The Obukhov length, friction velocity, surface pressure, 2-meter air temperature, relative humidity, and specific humidity are also included. In adition the turbulent kinetic energy at 50, 100, 150, 200 and 300m are provided with 3 hourly time step without resampling.
  
  .. code-block:: python

    product='NORA3_offshore_wind'

  Dataset: The original dataset are https://thredds.met.no/thredds/projects/nora3.html and https://thredds.met.no/thredds/catalog/nora3_subset_atmos/atm_3hourly/catalog.html. This work was done at SINTEF within the Horizon Europe `project WILLOW <https://www.sintef.no/en/projects/2023/willow/>`_ .

* For coastal wave NORA3 data developed by MET Norway: 

  .. code-block:: python
    
    product='NORAC_wave'

  Dataset: https://thredds.met.no/thredds/catalog/norac_wave/field/catalog.html

* For coastal wave spectra (Norwegian coast) developed by MET Norway: 

  .. code-block:: python
    
    product='NORAC_wave_spec'

  Dataset: https://thredds.met.no/thredds/catalog/norac_wave/spec/catalog.html

* For ocean data (sea level, temperature, currents, salinity over depth ) Norkyst800 data (from 2016-09-14 to today) developed by MET Norway: 

  .. code-block:: python

    product='NORKYST800'

  Dataset: https://thredds.met.no/thredds/fou-hi/norkyst800v2.html

* For ocean data (sea level, temperature, currents, salinity over depth ) NorkystDA data (for 2017-2018) developed by MET Norway: 

  .. code-block:: python
    
    product='NorkystDA_zdepth' or product='NorkystDA_surface' (for only surface data) 

  Dataset: https://thredds.met.no/thredds/catalog/nora3_subset_ocean/catalog.html

* For global reanalysis ERA5 (wind and waves) developed by ECMWF: 

  .. code-block:: python
  
    product='ERA5' 

  The user needs to install the *CDS API key* according to https://cds.climate.copernicus.eu/how-to-api ,
  Dataset: https://doi.org/10.24381/cds.adbb2d47

* For Global Tide and Surge Model (storm surge, tidal elevation, total water level) developed by Deltares : 

  .. code-block:: python

    product='GTSM' 

  The user needs to install the *CDS API key* according to https://cds.climate.copernicus.eu/api-how-to ,
  Dataset: https://doi.org/10.24381/cds.a6d42d60

* For ECHOWAVE (European COasts High Resolution Ocean WAVEs Hindcast) developed by Marine Renewable Energies Lab (MREL), TU Delft: 

  .. code-block:: python

    product='ECHOWAVE' 
    
  Dataset: https://doi.org/10.4121/f359cd0f-d135-416c-9118-e79dccba57b9.v1, Publication: https://doi.org/10.1016/j.renene.2024.121391

* For wave buoy observations (Statens vegvesen - E39): product='E39_letter_location_wave', e.g,  

  .. code-block:: python

    product='E39_B_Sulafjorden_wave'
 
  Dataset: https://thredds.met.no/thredds/catalog/obs/buoy-svv-e39/catalog.html

Import data 
=====================================
Import data from server to **ts-object** and save it as csv:

.. code-block:: python

   df_ts.import_data(save_csv=True, save_nc=False)

Data is saved in:

.. code-block:: python

   print(df_ts.datafile) #'NORA3_wind_wave_lon1.32_lat53.324_20000101_20000331.csv' 

To import data from a local csv-file to **ts-object**:

.. code-block:: python

   df_ts.load_data(local_file=df_ts.datafile)  
   print(df_ts.data)

.. image:: ts.data0.png
  :width: 900


Combine csv-files 
=====================================

To combine several csv-files produced by **metocean-api**:

.. code-block:: python

   df = ts.ts_mod.combine_data(list_files=['NORA3_wind_sub_lon1.32_lat53.324_20210101_20210131.csv',
                                           'NORA3_atm_sub_lon1.32_lat53.324_20210101_20210331.csv'],
                                   output_file='combined_NORA3_lon1.32_lat53.324.csv')  



Command Line Interface (CLI) - metocean-cli
===========================================

**Download Command**


  The ``download`` command is used to download metocean data for a specified product, location, and time range.

  **Usage:**

  .. code-block:: bash

    metocean-cli download <product> <lat> <lon> <start_time> <stop_time> <file_format> [--use_cache] [--max_retry MAX_RETRY] [-o OUTPUT]

  **Arguments:**

  - ``<product>``: Name of the product to download.
  - ``<lat>``: Latitude of the location.
  - ``<lon>``: Longitude of the location.
  - ``<start_time>``: Start time for data in ISO 8601 format (e.g., ``"2023-01-01"``).
  - ``<stop_time>``: Stop time for data in ISO 8601 format (e.g., ``"2023-01-02"``).
  - ``<file_format>``: Format of the output file: ``"csv"``, ``"netcdf"``, or ``"both"``.
  - ``--use_cache``: Optional flag to use cached data if available. Default is ``True``.
  - ``--max_retry MAX_RETRY``: Optional argument to specify the maximum number of retry attempts for the download. Default is ``5``.
  - ``-o OUTPUT``, ``--output OUTPUT``: Path to the output file where the downloaded data will be saved.

  **Example:**

  .. code-block:: bash

    metocean-cli download NORA3_offshore_wind 54.01 6.58 "2023-01-01" "2023-01-02" csv

**Combine Command**

  The ``combine`` command is used to combine multiple metocean data files into a single output file.

  **Usage:**

  .. code-block:: bash

    metocean-cli combine <files>... -o OUTPUT

  **Arguments:**

  - ``<files>...``: List of file paths to combine. You can specify multiple files separated by spaces.
  - ``-o OUTPUT``, ``--output OUTPUT``: Path to the output file where the combined data will be saved.

  **Example:**

  .. code-block:: bash

    metocean-cli combine 'NORA3_wind_sub_lon1.32_lat53.324_20210101_20210131.csv' 'NORA3_atm_sub_lon1.32_lat53.324_20210101_20210331.csv' -o combined.csv



.. toctree::
   :maxdepth: 2
   :caption: Contents:
