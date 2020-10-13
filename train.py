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
    parser.add_argument("--data", help="CSV file with training data")
    parser.add_argument("--epochs", type=int, help="Training epochs")
    parser.add_argument("--cuda_device", help="CUDA device number (pick one)")
    parser.add_argument("--batch", type=int, help="Number of samples taken for single evaluation")
    parser.add_argument("--population", type=int, help="Population size")
    parser.add_argument("--model_size", type=int, help="Model size")
    parser.add_argument("--max_evaluation_steps", type=int, help="Maximum number of evaluation steps")
    parser.add_argument("--take_profit", type=float, help="Take profit: 1.05 means +5%")
    parser.add_argument("--stop_loss", type=float, help="Stop loss: 0.95 means -5%")
    parser.add_argument("--save_dir", help="Model save directory")
    return parser.parse_args()


def main(args):
    data = data_stream(args.data, args.batch)
    if args.cuda_device:
        device = 'cuda'
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    else:
        device = 'cpu'
    population = Population(device, args)
    for epoch in range(args.epochs):
        best_result_so_far = population.train()
        population.save(args.save_dir)
        logging.info(f'Epoch: {epoch}; best trade result: {best_result_so_far}')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)s: %(message)s')
    main(parse_command_line_arguments())
