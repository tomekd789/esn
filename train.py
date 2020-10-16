"""
Train the ESN model

Example parameters:
    --data /opt/dane_synology/esn_data_100k.csv \
    --epochs 100 \
    --cuda_device 3 \
    --batch 200 \
    --population 10 \
    --model_size 5 \
    --mutation_probability 0.001 \
    --co_probability 0.4 \
    --max_evaluation_steps 10 \
    --take_profit 1.05 \
    --stop_loss 0.95 \
    --save_dir /home/tdryjanski/esn_model

Meaning of selected parameters:
    --data: path to a file containing two-week minute andromeda trade sequences normalized to start from 1.0
    --batch: number of random trade sequences applied to the model sequentially
        (we start with $1.00 and apply all the sequences one by one)
    --population: number of models evaluated in parallel
    --model_size: size of the model internal state (aka neuron count)
    --mutation_probability: after a new population is created as a copy of the existing one,
        weights and biases are modified randomly with the given probability.
        For mutation functions see model.MUTATION_FUNCTIONS, a random one is picked each time
    --co_probability: after mutations are done, models are mutually crossed-over with the given probability
    --max_evaluation_steps: max RNN steps allowed. If model does not take a decision during this time, no trade happens
    --take_profit, stop_loss: a model returns starting index in the sequence, and the 'buy' or 'sell' signal
        after that, a box trade is executed till the end of the sequence, with take profit and stop loss set
        The numbers are factorials (i.e. 1.05 means +5% price move, 0.95 means -5%, relative to the trade start price;
        actual loss or profit depends on the trade direction, later)
"""
import argparse
import logging
from math import exp, log
import os

from data import data_stream
from model import Population

WEEKS_PER_SEQUENCE = 2


def parse_command_line_arguments():  # pylint: disable=missing-function-docstring
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="CSV file with training data")
    parser.add_argument("--epochs", type=int, help="Training epochs")
    parser.add_argument("--cuda_device", help="CUDA device number (pick one)")
    parser.add_argument("--batch", type=int, help="Number of samples taken for single evaluation")
    parser.add_argument("--population", type=int, help="Population size")
    parser.add_argument("--model_size", type=int, help="Model size")
    parser.add_argument("--mutation_probability", type=float, help="Mutation probability")
    parser.add_argument("--co_probability", type=float, help="Crossing-over probability")
    parser.add_argument("--max_evaluation_steps", type=int, help="Maximum number of evaluation steps")
    parser.add_argument("--take_profit", type=float, help="Take profit: 1.05 means +5%")
    parser.add_argument("--stop_loss", type=float, help="Stop loss: 0.95 means -5%")
    parser.add_argument("--save_dir", help="Model save directory")
    return parser.parse_args()


def main(args):
    args_as_string = "\n    --" + str(args).replace("Namespace(", "").replace(")", "").replace(", ", "\n    --")
    args_as_string = args_as_string.replace("=", " ")
    logging.info(f"Training started; arguments: {args_as_string}")
    data = data_stream(args.data, args.batch)
    if args.cuda_device:
        device = 'cuda'
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    else:
        device = 'cpu'
    population = Population(device, data, args)
    for epoch in range(args.epochs):
        log_best_result_so_far, trades_count = population.train()
        # best_result_so_far is $s owned, from $1.00 after applying the batch sequence
        # we need to normalize to get the yearly percent profit
        trade_duration_weeks = args.batch * WEEKS_PER_SEQUENCE
        trade_duration_years = trade_duration_weeks / 52
        log_best_result_per_year = log_best_result_so_far / trade_duration_years
        wallet_after_a_year = exp(log_best_result_per_year)
        yearly_gain_percent = (wallet_after_a_year - 1.0) * 100  # Wallet value starts at $1.00
        population.save(args.save_dir)
        logging.info(f"Epoch: {epoch + 1}; " +
                     f"best evaluation: {exp(log_best_result_so_far):.2f}; " +
                     f"best model's yearly gain: {yearly_gain_percent:.1f}%; " +
                     f"trades: {trades_count} (of {args.batch} possible)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)s: %(message)s')
    main(parse_command_line_arguments())
