import argparse
import sys
import time
import os
from typing import Optional, List

from metocean_api import ts

class FileNotCreatedError(Exception):
    """Exception raised when a file is not created successfully."""
    def __init__(self, message="The file was not created successfully."):
        self.message = message
        super().__init__(self.message)

def download(product: str, lat: float, lon: float, start_time: str, stop_time: str,
             file_format: str, use_cache: bool = True, max_retry: int = 5,
             output_file: Optional[str] = None) -> None:
    """
    Download metocean data for a specified product, location, and time range.

    Args:
        product (str): The product to download.
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        start_time (str): Start time in ISO 8601 format.
        stop_time (str): Stop time in ISO 8601 format.
        file_format (str): File format to download ('csv', 'netcdf', or 'both').
        use_cache (bool): Whether to use cached data. Defaults to True.
        max_retry (int): Maximum number of retry attempts. Defaults to 5.
        output_file (Optional[str]): Path to the output file. Required for execution.
    """
    print(f"Downloading {product} data from {start_time} to {stop_time} at ({lat}, {lon}) in {file_format} format.")
    print(f"Use cache: {use_cache}, Max retry: {max_retry}")

    file_format_options = {
        'save_csv': False,
        'save_nc': False
    }

    if file_format == 'csv':
        file_format_options['save_csv'] = True
    elif file_format == 'netcdf':
        file_format_options['save_nc'] = True
    elif file_format == 'both':
        file_format_options['save_csv'] = True
        file_format_options['save_nc'] = True

    retry_count = 0
    success = False

    while retry_count < max_retry and not success:
        try:
            # Attempt to create TimeSeries object and import data
            df_ts = ts.TimeSeries(lon=lon, lat=lat,
                                  start_time=start_time, end_time=stop_time,
                                  product=product, datafile=output_file) # type: ignore

            output_file = df_ts.datafile

            df_ts.import_data(**file_format_options, use_cache=use_cache)
            success = True

        except Exception as e:
            print(f"An error occurred: {e}")
            retry_count += 1
            print(f"Retrying... Attempt {retry_count}/{max_retry}")
            time.sleep(5)  # Wait for a moment before retrying

    if not success:
        print("Failed to download data after several attempts.")
    else:
        # Verify if the output file was created
        if os.path.exists(output_file): # type: ignore
            print(f"Data successfully downloaded to {output_file}")
        else:
            print("Download process completed but output file was not created.")

def combine(files: List[str], output_file: str) -> None:
    """
    Combine multiple files into a single output file.

    Args:
        files (List[str]): List of file paths to combine.
        output_file (str): Path to the output file.
    """
    print(f"Combining files: {files} into {output_file}")
    df = ts.ts_mod.combine_data(list_files=files, output_file=output_file)

def main():
    parser = argparse.ArgumentParser(
        prog=f'metocean-cli',
        description='Metocean-api CLI: A command-line tool to extract time series of metocean data from global/regional/coastal hindcasts/reanalysis',
        epilog='''
    MET-OM/metocean-api Extract time series of metocean data from global/regional/coastal hindcasts/reanalysis
    Copyright (C) <year>  KonstantinChri

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
    USA
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')

    # Download command
    download_parser = subparsers.add_parser('download', help='Download metocean time series data')
    download_parser.add_argument('product', type=str, help='Name of the product to download. The list of the product is available at https://metocean-api.readthedocs.io/en/latest/#available-datasets-in-metocean-api')
    download_parser.add_argument('lat', type=float, help='Latitude of the location')
    download_parser.add_argument('lon', type=float, help='Longitude of the location')
    download_parser.add_argument('start_time', type=str, help='Start time for data in ISO 8601 format')
    download_parser.add_argument('stop_time', type=str, help='Stop time for data in ISO 8601 format')
    download_parser.add_argument('file_format', type=str, choices=['csv', 'netcdf', 'both'],
                                 help='Format of the output file: "csv", "netcdf", or "both"')
    download_parser.add_argument('--no_cache', default=False,
                                 help='Clear cached data at the end of the processing - not recommanded in case of faillure or large dataset')
    download_parser.add_argument('--max_retry', type=int, default=5,
                                 help='Maximum number of retry attempts for the download')
    download_parser.add_argument('-o', '--output', type=str, default=None, required=False,
                                 help='Path to the output file')

    # Combine command
    combine_parser = subparsers.add_parser('combine', help='Combine multiple metocean data files generated using metocean-api')
    combine_parser.add_argument('files', type=str, nargs='+',
                                help='List of file paths to combine')
    combine_parser.add_argument('-o', '--output', type=str, required=True,
                                help='Path to the output combined file')

    args = parser.parse_args()

    if args.command == 'download':
        download(args.product, args.lat, args.lon, args.start_time, args.stop_time,
                 args.file_format, not args.no_cache, args.max_retry, args.output)
    elif args.command == 'combine':
        combine(args.files, args.output)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
