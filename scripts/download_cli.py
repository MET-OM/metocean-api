import argparse
import sys

from metocean_api import ts


def download(product : str, 
             lat : float, lon : float,
             start_time : str, stop_time : str, file_format : str,
             use_cache : bool =True, max_retry : int = 5):
    
    print(f"Downloading {product} data from {start_time} to {stop_time} at ({lat}, {lon}) in {file_format} format.")
    print(f"Use cache: {use_cache}, Max retry: {max_retry}")


    file_format_options = {
        'csv': False,
        'netcdf': False
    }

    if file_format == 'csv':
        file_format_options['csv'] = True
    elif file_format == 'netcdf':
        file_format_options['netcdf'] = True
    elif file_format == 'both':
        file_format_options['csv'] = True
        file_format_options['netcdf'] = True

    try:
        df_ts = ts.TimeSeries(lon=lon, lat=lat,
                    start_time=start_time, end_time=start_time ,
                    product=product, datafile=)
        
        df_ts.import_data(**file_format_options, use_cache=use_cache)





def combine(files : list[str], output_file : str):
    # Your combine logic here
    print(f"Combining files: {files} into {output_file}")
    df = ts.ts_mod.combine_data(list_files=files,
                                output_file=output_file)

def main():
    parser = argparse.ArgumentParser(description='Metocean-api CLI')
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')

    # Download command
    download_parser = subparsers.add_parser('download', help='Download metocean data')
    download_parser.add_argument('product', type=str, help='Product to download')
    download_parser.add_argument('lat', type=float, help='Latitude')
    download_parser.add_argument('lon', type=float, help='Longitude')
    download_parser.add_argument('start_time', type=str, help='Start time in ISO 8601 format')
    download_parser.add_argument('stop_time', type=str, help='Stop time in ISO 8601 format')
    download_parser.add_argument('file_format', type=str, choices=['csv', 'netcdf', 'both'], help='File format')
    download_parser.add_argument('--use_cache', type=bool, default=True, help='Use cache')
    download_parser.add_argument('--max_retry', type=int, default=5, help='Maximum number of retries')
    download_parser.add_argument('-o', '--output', type=str, required=True, help='Output file')

    # Combine command
    combine_parser = subparsers.add_parser('combine', help='Combine multiple files')
    combine_parser.add_argument('files', type=str, nargs='+', help='Files to combine')
    combine_parser.add_argument('-o', '--output', type=str, required=True, help='Output file')

    args = parser.parse_args()

    if args.command == 'download':
        download(args.product, args.lat, args.lon, args.start_time, args.stop_time, args.file_format, args.use_cache, args.max_retry, args.output)
    elif args.command == 'combine':
        combine(args.files, args.output)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
