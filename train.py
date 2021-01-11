"""
Train the ESN model

Example parameters:
    --id model_id
    --recover no
    --data_url http://127.0.0.1:5000/sequence \
    --epochs 100 \
    --cuda_device 3 \
    --batch 200 \
    --population 10 \
    --model_size 5 \
    --mutation_probability 0.001 \
    --co_probability 0.4 \
    --esn_input_size 20 \
    --max_evaluation_steps 10 \
    --take_profit 1.05 \
    --stop_loss 0.95 \
    --save_dir /home/tdryjanski/esn_model

Meaning of selected parameters:
    --data_url: see "Flask service" in README.md
    --sequences: number of random trade sequences applied to the model sequentially but independently
        (Each sequence starts at $1.0)
    --population: number of models evaluated in parallel
    --model_size: size of the model internal state (aka neuron count)
    --mutation_probability: after a new population is created as a copy of the existing one,
        weights and biases are modified randomly with the given probability.
        For mutation functions see model.MUTATION_FUNCTIONS, a random one is picked each time
    --co_probability: after mutations are done, models are mutually crossed-over with the given probability
    --esn_input_size: size of ESN parallel input (count of subsequent prices from the input sequence processed at once)
    --max_evaluation_steps: max RNN steps allowed. If model does not take a decision during this time, no trade happens
    --take_profit, stop_loss: a model returns starting index in the sequence, and the 'buy' or 'sell' signal
        after that, a box trade is executed till the end of the sequence, with take profit and stop loss set
        The numbers are factorials (i.e. 1.05 means +5% price move, 0.95 means -5%, relative to the trade start price;
        actual loss or profit depends on the trade direction, later)
"""
import argparse
import logging
import os

from model import Population

WEEKS_PER_SEQUENCE = 2


def parse_command_line_arguments():  # pylint: disable=missing-function-docstring
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", help="Experiment ID (a user-defined string)")
    parser.add_argument("--recover", help="Recover from saved checkpoint (yes/no; default no)")
    parser.add_argument("--data_url", help="REST server for training data")
    parser.add_argument("--epochs", type=int, help="Training epochs")
    parser.add_argument("--cuda_device", help="CUDA device number (pick one)")
    parser.add_argument("--batch", type=int, help="Number of samples taken for single evaluation")
    parser.add_argument("--population", type=int, help="Population size")
    parser.add_argument("--model_size", type=int, help="Model size")
    parser.add_argument("--mutation_probability", type=float, help="Mutation probability")
    parser.add_argument("--co_probability", type=float, help="Crossing-over probability")
    parser.add_argument("--esn_input_size", type=int, help="ESN input size")
    parser.add_argument("--max_evaluation_steps", type=int, help="Maximum number of evaluation steps")
    parser.add_argument("--take_profit", type=float, help="Take profit: 1.05 means +5%")
    parser.add_argument("--stop_loss", type=float, help="Stop loss: 0.95 means -5%")
    parser.add_argument("--save_dir", help="Model save directory")
    return parser.parse_args()


def main(args):  # pylint: disable=missing-function-docstring
    args_as_string = "\n    --" + str(args).replace("Namespace(", "").replace(")", "").replace(", ", "\n    --")
    args_as_string = args_as_string.replace("=", " ").replace("'", "")
    args_as_string += "\n"
    logging.info("Training started; arguments: %s", args_as_string)
    if args.cuda_device:
        device = 'cuda'
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    else:
        device = 'cpu'
    population = Population(device, args, args_as_string)
    for epoch in range(args.epochs):
        best_result_so_far, trades_count = population.train()
        # best_result_so_far is $s loss/gain from $1.00, summed by the batch sequence
        # we need to normalize to get the yearly percent profit
        trade_duration_weeks = args.batch * WEEKS_PER_SEQUENCE
        trade_duration_years = trade_duration_weeks / 52
        best_result_per_year = best_result_so_far / trade_duration_years
        yearly_gain_percent = best_result_per_year * 100  # Cash starts at $1.00
        population.save(args.save_dir)
        logging.info(f"Epoch: {epoch + 1}; " +
                     f"best evaluation: {best_result_so_far:.2f}; " +
                     f"best model's yearly gain: {yearly_gain_percent:.1f}%; " +
                     f"trades: {trades_count} (of {args.batch} possible); " +
                     f"model id: {population.id}"
                     )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)s: %(message)s')
    main(parse_command_line_arguments())
