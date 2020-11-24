"""
Utils for handling the Andromeda universe
"""
import os

TWO_WEEKS = 5865  # of minute tickers; checked empirically ;)


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
