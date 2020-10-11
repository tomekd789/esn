"""
Train the ESN model
"""
import argparse
import logging
import os

from data import data_stream
from model import Population


def parse_command_line_arguments():  # pylint: disable=missing-function-docstring
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="CSV file with training data")
    parser.add_argument("-e", "--epochs", type=int, help="Training epochs")
    parser.add_argument("-c", "--cuda_device", help="CUDA device number (pick one)")
    parser.add_argument("-b", "--batch", type=int, help="Number of samples taken for single evaluation")
    parser.add_argument("-p", "--population", type=int, help="Population size")
    parser.add_argument("-m", "--model_size", type=int, help="Model size")
    parser.add_argument("-s", "--save", help="Model save path")
    return parser.parse_args()


def main(args):
    data = data_stream(args.data, args.batch)
    if args.cuda_device:
        device = 'cuda'
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    else:
        device = 'cpu'
    population = Population(device, args.population, args.model_size, data)
    for epoch in range(args.epochs):
        best_result_so_far = population.train()
        population.save(args.save)
        logging.info(f'Epoch: {epoch}; best trade result: {best_result_so_far}')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)s: %(message)s')
    main(parse_command_line_arguments())
