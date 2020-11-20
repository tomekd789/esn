"""
Flask service for providing random trade sequences, normalized
It returns a random sequence normalized to start from 1.0 value from a random andromeda ticker

See example_for_framework.py for usage details
"""
import argparse
import flask
import os
import random

from andromeda_utils import get_all_andromeda_tickers

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    return "<h1>Andromeda data service</h1><p>Returns a random two-week sequence from Andromeda 450, normalized.</p>"


@app.route('/sequence', methods=['GET'])
def get_sequence():
    while True:
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
        if ticker_length >= args.sequence_len * 2:  # Sequence is long enough; * 2 to avoid leakages
            break
    start_point = random.randint(0, ticker_length - args.sequence_len)
    sample = trade_data_buffer[ticker][start_point: start_point + args.sequence_len]
    # NORMALIZE the sample, so that the first trade value is 1.0
    trade_start_value = sample[0]
    normalized_sample = [value / trade_start_value for value in sample]
    return flask.jsonify(normalized_sample)


def parse_command_line_arguments():  # pylint: disable=missing-function-docstring
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--andromeda", help="Path to andromeda data folder")
    parser.add_argument("-y", "--year_since", type=int, help="Starting year")
    parser.add_argument("-l", "--sequence_len", type=int, help="Sequence length")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_command_line_arguments()
    all_andromeda_tickers = get_all_andromeda_tickers(args.andromeda)
    trade_data_buffer = {}  # Buffer to accumulate data in memory for future reuse
    app.run()
