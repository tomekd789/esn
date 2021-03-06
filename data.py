"""
Classes to handle locally cached data stream
Now made obsolete by the flask service
"""
import numpy as np  # pylint: disable=import-error
import pandas as pd  # pylint: disable=import-error


def basic_data_stream(data_file):
    """
    Infinite stream of samples taken from the data_file
    :param data_file: Path to the data file
    :return: sequence of data samples
    """
    dataset_pd = pd.read_csv(data_file)
    dataset = np.array(dataset_pd, dtype='float32')
    while True:
        # Iterate the data file and yield records one by one
        for index in range(dataset.shape[0]):
            yield dataset[index]
        # After the iteration is finished, reshuffle
        np.random.shuffle(dataset)


def data_stream(data_file, batch_size):
    """
    Generates infinite stream of data batches taken from the data_file
    :param data_file: Path to the data file
    :param batch_size: Data batch size
    :return: Random data batch
    """
    _basic_data_stream = basic_data_stream(data_file)
    while True:
        data_batch = [_basic_data_stream.__next__() for _ in range(batch_size)]
        yield data_batch
