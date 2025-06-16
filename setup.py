#!/usr/bin/env python

import os
import setuptools

here = os.path.abspath(os.path.dirname(__file__))
exec(open(os.path.join(here, 'version.py')).read())

setuptools.setup(
    name        = 'metocean-api',
    description = 'metocean-api - Tool to extract time series of metocean data from hindcasts/reanalysis',
    author      = 'Konstantinos Christakos MET Norway & NTNU',
    url         = 'https://github.com/MET-OM/metocean-api',
    download_url = 'https://github.com/MET-OM/metocean-api',
    version = __version__,
    license = 'LGPLv2.1',
    install_requires = [
        'matplotlib>=3.1',
        'numpy>=1.17',
        'pandas',
        'xarray',
        'dask',
        'pip',
        'netcdf4',
        'cartopy',
        'metpy',
        'tqdm'
    ],
    packages = setuptools.find_packages(),
    include_package_data = True,
    setup_requires = ['setuptools_scm'],
    tests_require = ['pytest','pytest-cov'],
    scripts = [],
    entry_points={
        'console_scripts': [
            'metocean-cli=metocean_api.cli.metocean_cli:main',
        ],
    },
)


