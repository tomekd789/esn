"""
Run inference on a single ESN model

Example parameters:
    --data /opt/dane_synology/esn_data_100k.csv \
    --sequences 200 \
    --max_evaluation_steps 10 \
    --take_profit 1.05 \
    --stop_loss 0.95 \
    --load_dir /home/tdryjanski/esn_model
    --file_prefix 20201016145331

Meaning of selected parameters:
    --data: path to a file containing two-week minute andromeda trade sequences normalized to start from 1.0
    --sequences: number of random trade sequences applied to the model sequentially
        (we start with $1.00 and apply all the sequences one by one)
    --max_evaluation_steps: max RNN steps allowed. If model does not take a decision during this time, no trade happens
    --take_profit, stop_loss: a model returns starting index in the sequence, and the 'buy' or 'sell' signal
        after that, a box trade is executed till the end of the sequence, with take profit and stop loss set
        The numbers are factorials (i.e. 1.05 means +5% price move, 0.95 means -5%, relative to the trade start price;
        actual loss or profit depends on the trade direction, later)
    --file_prefix: time stamp in the saved model files name
"""
import argparse
import logging
from math import exp, log
import os

from data import basic_data_stream
from model import Model

WEEKS_PER_SEQUENCE = 2


def parse_command_line_arguments():  # pylint: disable=missing-function-docstring
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="CSV file with training data")
    parser.add_argument("--sequences", type=int, help="Number of samples taken for single evaluation")
    parser.add_argument("--max_evaluation_steps", type=int, help="Maximum number of evaluation steps")
    parser.add_argument("--take_profit", type=float, help="Take profit: 1.05 means +5%")
    parser.add_argument("--stop_loss", type=float, help="Stop loss: 0.95 means -5%")
    parser.add_argument("--load_dir", help="Model save directory")
    parser.add_argument("--file_prefix", help="File names prefix")
    return parser.parse_args()


def main(args):
    data = basic_data_stream(args.data)
    device = 'cpu'
    model = Model(device, data, args)
    gain_so_far = 0.0
    for sequence_counter in range (args.sequences):
        sequence = data.__next__()
        gain, trade_start_pointer = model.evaluate_sequence(sequence)
        gain_so_far += gain
        trade_duration_weeks = sequence_counter * WEEKS_PER_SEQUENCE
        trade_duration_years = trade_duration_weeks / 52
        yearly_gain = gain_so_far / trade_duration_years
        yearly_gain_percent = yearly_gain * 100
        if sequence_counter % 1000 == 0:
            logging.info(f"Sequence {sequence_counter + 1}; " +
                         f"yearly gain: {yearly_gain_percent:.2f}%; " +
                         f"trade start point: {trade_start_pointer} (of {len(sequence)} possible)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)s: %(message)s')
    main(parse_command_line_arguments())
