"""
Flask service for providing trade prediction for a sequence

Returns
"""
import argparse
from flask import Flask, request, jsonify

from model import Model, get_rest_data

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    return "<h1>ESN Predictor</h1><p>Returns a trade prediction given a sequence.</p>"


@app.route('/predict', methods=['POST'])
def predict():
    sequence = request.json
    if len(sequence) < model.esn_input_size:
        return "Input sequence to short\n", 400
    # Normalize sequence if needed
    start_price = sequence[0]
    if start_price <= 0:
        return "Initial price cannot be zero\n", 400
    if start_price != 1.0:
        sequence = [price / start_price for price in sequence]
    trade, start_point = model.infer(sequence)
    return jsonify([trade, start_point])


def parse_command_line_arguments():  # pylint: disable=missing-function-docstring
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", help="Model ID string")
    parser.add_argument("--cuda_device", help="CUDA device number (pick one)")
    parser.add_argument("--esn_input_size", type=int, help="ESN input size")
    parser.add_argument("--max_evaluation_steps", type=int, help="Maximum number of evaluation steps")
    parser.add_argument("--take_profit", type=float, help="Take profit: 1.05 means +5%")
    parser.add_argument("--stop_loss", type=float, help="Stop loss: 0.95 means -5%")
    parser.add_argument("--load_dir", help="Model load directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_command_line_arguments()
    device = 'cpu'
    model = Model(device, args)
    app.run(port = 5100)
