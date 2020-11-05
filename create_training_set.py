"""
Randomly takes two-week trade sequences from Andromeda and saves in a file

CLI parameters:
    "-a", "--andromeda": path to the andromeda files storage; I take <TICKER>_full_data.csv files from there
    "-s", "--samples", type=int: number of selected data samples. Data sample is a random two-week period of trade
        prices starting on Jan 1st 00:00 of the --year_since parameter value
    "-y", "--year_since": starting year; older trades are disregarded
    "-t", "--target": target file for saving samples

Example:
    --andromeda /opt/dane/ssd/data_cache/andromeda \
    --year_since 2017 \
    --samples 100000 \
    --target /opt/dane_synology/esn_data_100k.csv.csv

Memory demand:
    memory cache for data since 2017 -> 11 GB RAM
    100k samples: 3.4 G data file
"""
import argparse
import csv
import logging
import os
import random

from algo_logger import logger

TWO_WEEKS = 5865  # of minute tickers; checked empirically ;)


def parse_command_line_arguments():  # pylint: disable=missing-function-docstring
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--andromeda", help="Path to andromeda data folder")
    parser.add_argument("-s", "--samples", type=int, help="Sample size")
    parser.add_argument("-y", "--year_since", type=int, help="Starting year")
    parser.add_argument("-t", "--target", help="target file")
    return parser.parse_args()


def get_all_andromeda_tickers(andromeda_path):
    """
    List all tickers in the given path, unique
    :param andromeda_path: Path to andromeda files
    :return: List of all ticker files found there, unique
    """
    all_files_in_the_directory = os.listdir(andromeda_path)
    # Look for the first parts of the names (till '_') and make set of them to remove duplicates
    tickers = {file_name.split("_")[0] for file_name in all_files_in_the_directory}
    # Change the set to a list
    return list(tickers)


def main(args):  # pylint: disable=missing-function-docstring
    all_andromeda_tickers = get_all_andromeda_tickers(args.andromeda)
    generated_samples_counter = 0
    progress_logger = logger(frequency=100, maximum=args.samples)
    trade_data_buffer = {}  # Buffer to accumulate data in memory for future reuse
    with open(args.target, 'w') as target_file_handle:
        target_file_writer = csv.writer(target_file_handle, quoting=csv.QUOTE_MINIMAL)
        while generated_samples_counter < args.samples:
            progress_logger()
            ticker = random.choice(all_andromeda_tickers)  # Get a random ticker
            # Populate the trade data buffer with ticker data if not yet exists
            if ticker not in trade_data_buffer.keys():
                ticker_file = os.path.join(args.andromeda, ticker + '_full_data.csv')
                trade_data_buffer[ticker] = []
                with open(ticker_file) as ticker_file_handle:
                    for line in ticker_file_handle:
                        try:
                            trade_year = int(line[:4])
                        except ValueError:  # Ignore header line
                            continue
                        if trade_year >= args.year_since:
                            _, trade_value = line.strip().split(',')
                            trade_value = float(trade_value)
                            trade_data_buffer[ticker].append(trade_value)
            ticker_length = len(trade_data_buffer[ticker])
            if ticker_length < TWO_WEEKS * 2:  # Sequence is too short to sample from; * 2 to avoid leakages
                continue
            start_point = random.randint(0, ticker_length - TWO_WEEKS)
            sample = trade_data_buffer[ticker][start_point: start_point + TWO_WEEKS]
            # NORMALIZE the sample, so that the first trade value is 1.0
            trade_start_value = sample[0]
            normalized_sample = [value / trade_start_value for value in sample]
            target_file_writer.writerow(normalized_sample)
            generated_samples_counter += 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)s: %(message)s')
    main(parse_command_line_arguments())
