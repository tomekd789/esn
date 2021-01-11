"""
Classes for serving a models' population and a single model
"""
import json
import logging
import os
import random

import requests
import torch  # pylint: disable=import-error

MUTATION_FUNCTIONS = [
    lambda x: x / random.uniform(0.999, 1.001),  # Small gradient-like step
    lambda x: x + random.uniform(-1.0, 1.0)  # Leap
]

MAX_MUTATION_PROBABILITY = 0.01


def _get_mutation_function(mutation_probability):
    """
    Choose a mutation function randomly from the given list
    :param mutation_probability: with 1 - mutation probability return identity (no mutation)
    :return: float -> float function executing the mutation (also identity in case of no mutation)
    """
    if random.random() > mutation_probability:
        return lambda x: x
    return random.choice(MUTATION_FUNCTIONS)


def calculate_trade_outcome(sequence, trade_start_pointer, trade_type, take_profit, stop_loss):
    """
    Calculate the trade outcome for a given prices sequence, moment of start, and position (long or short)
    starting with 1.0 cash.
    Note: if trade pointer is at the end of the sequence, the trade is not performed
    :param sequence: asset prices in the sequential order
    :param trade_start_pointer: the pointer where the trade actually begins
    :param trade_type: can be "long" or "short", for buy or sell transactions
    :param take_profit: multiplier for T/P, e.g. 1.05 means +5%
    :param stop_loss: multiplier for S/L, e.g. 0.95 means -5%
    :return: gain after trade, bool information if any trade was actually performed
    """
    # We start this calculation with $1.00 and buy at the trade start price
    # To simulate continuous trading, we multiply the outcomes from single sequences
    if trade_start_pointer >= len(sequence):
        return 0.0, False  # No loss/gain, trade not executed
    trade_start_price = sequence[trade_start_pointer]
    purchased_stocks = 1.0 / trade_start_price
    sequence_index = trade_start_pointer + 1
    tp_closing_price = trade_start_price * take_profit
    sl_closing_price = trade_start_price * stop_loss
    while sequence_index < len(sequence) - 1:
        # Check for Take Profit
        if sequence[sequence_index] >= tp_closing_price:
            break
        # Check for Stop Loss
        if sequence[sequence_index] <= sl_closing_price:
            break
        sequence_index += 1
    trade_close_price = sequence[sequence_index]
    result = purchased_stocks * trade_close_price
    gain = result - 1.0
    if trade_type == "short":
        gain = -gain
        gain = max(gain, -1.0)  # I assume that we cannot loose more than we had (is this correct?)
    return gain, True


def get_rest_data(data_url, batch_size):
    """
    Reach the data provider service and collect a batch of sequences
    :param data_url: data provider service URL
    :param batch_size: number of sequences to return
    :return: batch-sized list of sequences as lists of floats
    """
    return [json.loads(requests.get(data_url).text) for _ in range(batch_size)]


class Population:  # pylint: disable=too-many-instance-attributes
    """
    Implementation of the population of models, with methods to generate and evaluate them
    """
    def __init__(self, device, args, args_as_string):
        self.id = args.id  # pylint: disable=invalid-name
        self.device = device
        self.args = args
        self.args_as_string = args_as_string
        self.population_size = args.population
        self.model_size = args.model_size
        self.mutation_probability = min(args.mutation_probability, MAX_MUTATION_PROBABILITY)
        self.co_probability = args.co_probability
        self.esn_input_size = args.esn_input_size
        self.max_evaluation_steps = args.max_evaluation_steps
        self.take_profit = args.take_profit
        self.stop_loss = args.stop_loss
        # Zero initialization
        self.population_weights = torch.zeros([self.population_size, self.model_size, self.model_size], device=device)
        self.population_biases = torch.zeros([self.population_size, self.model_size], device=device)
        self.population_evaluations = torch.zeros(self.population_size)
        self.new_population_weights = None
        self.new_population_biases = None
        self.data_url = args.data_url
        self.batch = args.batch
        self.trades_counters = [0] * self.population_size
        self.new_population_evaluations = None
        self.new_trades_counters = None
        # Recovery from checkpoint; weights, biases, and current evaluations are overwritten with saved values
        recover = args.recover
        if recover:
            recover = recover.upper()
            assert recover in ['YES', 'NO']
        else:
            recover = "NO"
        if recover == "YES":
            parameters_file_name = os.path.join(args.save_dir, self.id + '_checkpoint_parameters.json')
            weights_file_name = os.path.join(args.save_dir, self.id + '_weights.pt')
            biases_file_name = os.path.join(args.save_dir, self.id + '_biases.pt')
            evaluations_file_name = os.path.join(args.save_dir, self.id + '_evaluations.json')
            with open(parameters_file_name) as parameters_file:
                checkpoint_parameters = json.load(parameters_file)
            if args.population != checkpoint_parameters["population_size"] \
                    or args.model_size != checkpoint_parameters["model_size"] \
                    or args.esn_input_size != checkpoint_parameters["esn_input_size"]:
                raise ValueError("Checkpoint parameters mismatch")
            self.population_weights = torch.load(weights_file_name)
            self.population_biases = torch.load(biases_file_name)
            self.population_evaluations = torch.load(evaluations_file_name)

    def new_generation(self):
        """
        Takes a copy of the existing population and applies mutations and crossing-over to it
        :return: None
        """
        self._copy_population()
        self._mutate_new_population()
        self._crossing_over_in_new_population()

    def _copy_population(self):
        """
        Create a copy of the existing population; it will be randomly modified and then evaluated
        :return: None
        """
        self.new_population_weights = self.population_weights.clone()
        self.new_population_biases = self.population_biases.clone()

    def _mutate_new_population(self):
        """
        Mutate new population randomly
        Note: this code does not need to be very efficient, it will not be run frequently
        :return: None
        """
        for i in range(self.population_size):
            for j in range(self.model_size):
                for k in range(self.model_size):
                    mutation_function = _get_mutation_function(self.mutation_probability)
                    value_before_mutation = self.new_population_weights[i, j, k]
                    self.new_population_weights[i, j, k] = mutation_function(value_before_mutation)
        for i in range(self.population_size):
            for j in range(self.model_size):
                mutation_function = _get_mutation_function(self.mutation_probability)
                value_before_mutation = self.new_population_biases[i, j]
                self.new_population_biases[i, j] = mutation_function(value_before_mutation)

    def _crossing_over_in_new_population(self):
        """
        Cross-over models randomly
        :return: None
        """
        model_size_squared = self.model_size * self.model_size
        # Create a reshaped view of new population weights; values are shared, so we can do in situ modifications
        weights_reshaped = self.new_population_weights.view(self.population_size, model_size_squared)
        # Sample all pairs from the new population randomly
        crossing_over_already_done = [False] * self.population_size  # Flags to note if c.o. happened
        while True:
            remaining_population = [population_index for population_index in range(self.population_size)
                                    if not crossing_over_already_done[population_index]]
            if len(remaining_population) < 2:
                break
            first_index, second_index = random.sample(remaining_population, 2)
            crossing_over_already_done[first_index] = True
            crossing_over_already_done[second_index] = True
            if random.random() < self.co_probability:  # Actual crossing over happens here
                # Weights:
                # Select the split point; at least one element from either side needs to be taken
                co_split_index = random.choice(range(1, model_size_squared - 1))
                # Swap sequences
                saved_left_sequence = weights_reshaped[first_index][:co_split_index]
                saved_right_sequence = weights_reshaped[second_index][co_split_index:]
                weights_reshaped[first_index][:co_split_index] = weights_reshaped[second_index][:co_split_index]
                weights_reshaped[second_index][co_split_index:] = weights_reshaped[first_index][co_split_index:]
                weights_reshaped[second_index][:co_split_index] = saved_left_sequence
                weights_reshaped[first_index][co_split_index:] = saved_right_sequence
                # Biases (the same comments apply)
                # Align the biases split point roughly with the weights split point
                co_split_index = int(co_split_index / self.model_size)
                if co_split_index == 0:
                    co_split_index = 1  # At least one item to swap
                # It's easier to duplicate... I assume this code is written once anyway :)
                saved_left_sequence = self.new_population_biases[first_index][:co_split_index]
                saved_right_sequence = self.new_population_biases[second_index][co_split_index:]
                self.new_population_biases[first_index][:co_split_index] = \
                    self.new_population_biases[second_index][:co_split_index]
                self.new_population_biases[second_index][co_split_index:] = \
                    self.new_population_biases[first_index][co_split_index:]
                self.new_population_biases[second_index][:co_split_index] = saved_left_sequence
                self.new_population_biases[first_index][co_split_index:] = saved_right_sequence

    def _evaluate_sequence(self, weights, biases, sequence):  # pylint: disable=too-many-locals
        """
        Evaluate a single sequence
        :param weights: population models' weights
        :param biases: population models' biases
        :param sequence: single trade sequence as a python list
        :return: population-sized tensor of trade results, vector of bool flags showing if trade was actually performed
        """
        # Array of internal states for each model, zero-initialized
        internal_states = torch.zeros(self.population_size, self.model_size, device=self.device)
        # Models may have current sequence pointers set individually
        sequence_pointers = torch.zeros(self.population_size, device=self.device)
        # Flags to mark if models predict a trade (individually)
        is_trade = torch.zeros(self.population_size, dtype=torch.bool, device=self.device)
        # Pointers for trade start
        trade_start_pointers = torch.zeros(self.population_size, device=self.device)
        # Flags to take long position (buy)
        take_long_position = torch.zeros(self.population_size, dtype=torch.bool, device=self.device)
        # Flags to take long position (buy)
        take_short_position = torch.zeros(self.population_size, dtype=torch.bool, device=self.device)
        # Vectors used for masking
        all_true = torch.ones(self.population_size, dtype=torch.bool, device=self.device)
        all_zeros = torch.zeros(self.population_size, device=self.device)
        all_ones = torch.ones(self.population_size, device=self.device)
        # Maximum sequence pointer stored
        sequence_max_pointer = len(sequence) - self.esn_input_size
        # Perform RNN steps with all models in parallel (leverage the CUDA SIMD architecture)
        relu = torch.nn.ReLU()
        for _ in range(self.max_evaluation_steps):
            # For each internal state assign the first self.esn_input_size elements with the sequence values
            # according to sequence pointers
            pointers = sequence_pointers.tolist()
            pointers = [min(pointer, sequence_max_pointer) for pointer in pointers]
            for model_index, pointer in enumerate(pointers):
                for sequence_feed_index in range(self.esn_input_size):
                    internal_states[model_index][sequence_feed_index] =\
                        sequence[int(pointer) + sequence_feed_index]

            #   -----------------------------------------------------------------------------------------   #
            #                    THE CORE OPERATION.  Efficiency is much in demand here                     #
            #   -----------------------------------------------------------------------------------------   #
            # Scalar multiply internal states by corresponding models,
            # Option 1: the Einstein summation convention abstraction
            internal_states = torch.einsum("bn, bmn -> bm", internal_states, weights)
            # Option 2: explicit left side vector multiplication; no apparent speed gain
            # internal_states = torch.bmm(internal_states.unsqueeze(1), weights.transpose(2, 1)).squeeze(1)
            # Option 3: explicit right side vector multiplication; no apparent speed gain
            # internal_states = torch.bmm(weights, internal_states.unsqueeze(2)).squeeze(2)
            # add biases,
            internal_states += biases
            # and do the ReLU (important!)
            internal_states = relu(internal_states)
            #                                           (till here)                                         #

            # The last three state values have a special meaning:
            #     s[-1]: progress input pointer
            #     s[-2]: take long position (buy)
            #     s[-3] : take short position (sell)
            progress_input = internal_states[:, -1] >= 1.0
            buy_signals = internal_states[:, -2] >= 1.0
            sell_signals = internal_states[:, -3] >= 1.0
            # Buy and sell signals mutually cancel out
            collisions = torch.logical_and(buy_signals, sell_signals)
            buy_signals = torch.logical_and(buy_signals, torch.logical_not(collisions))
            sell_signals = torch.logical_and(sell_signals, torch.logical_not(collisions))
            # Exclude models with trade signal already on
            buy_signals = torch.logical_and(buy_signals, torch.logical_not(is_trade))
            sell_signals = torch.logical_and(sell_signals, torch.logical_not(is_trade))
            # Update long/short signals *if* the signal is set (otherwise do not change!)
            take_long_position = torch.where(buy_signals, all_true, take_long_position)
            take_short_position = torch.where(sell_signals, all_true, take_short_position)
            # The same for trade signals
            is_trade = torch.where(buy_signals, all_true, is_trade)
            is_trade = torch.where(sell_signals, all_true, is_trade)
            # Update trade start pointers *if* buy or sell signal is on
            trade_start_pointers = torch.where(buy_signals, sequence_pointers, trade_start_pointers)
            trade_start_pointers = torch.where(sell_signals, sequence_pointers, trade_start_pointers)
            # Update progress input
            step_pointers_forward = torch.where(progress_input, all_ones, all_zeros)
            sequence_pointers += step_pointers_forward * self.esn_input_size
            # Terminate the calculations if all trade signals have been switched on
            if is_trade.all():
                break
        # Calculate trades gains, model by model
        trades_results = torch.zeros(self.population_size, device=self.device)
        trades_executed = [False] * self.population_size
        for model_index in range(self.population_size):
            # If no trade was signalled, keep the existing value, i.e. 0.0
            if not is_trade[model_index]:
                continue
            trade_start_pointer = int(trade_start_pointers[model_index].item()) + self.esn_input_size
            trade_type = "long" if take_long_position[model_index] else "short"
            trade_outcome, trade_executed = calculate_trade_outcome(
                sequence, trade_start_pointer, trade_type, self.take_profit, self.stop_loss)
            trades_results[model_index] = trade_outcome
            trades_executed[model_index] = trade_executed
        return trades_results, trades_executed

    def _evaluate_population(self, weights, biases, batch):
        """
        The main evaluation method; takes a population (as weights & biases), the data stream,
        and returns trade values for models in the population
        :param weights: array of population models' weights
        :param biases: array of biases
        :return: population-sized list of trade results, list of trade counts actually executed
        """
        # Note that batch elements are processed sequentially; the name might be counterintuitive for DL practitioners
        # The calculations are rather batched along the population dimension, each model being evaluated independently
        accumulated_evaluation = torch.zeros(self.population_size, device=self.device)
        trades_executed_counters = [0] * self.population_size
        for sequence in batch:
            sequence_evaluation, trades_executed = self._evaluate_sequence(weights, biases, sequence)
            accumulated_evaluation += sequence_evaluation
            for i, trade_executed in enumerate(trades_executed):
                trades_executed_counters[i] += 1 if trade_executed else 0
        return accumulated_evaluation.tolist(), trades_executed_counters

    def _merge_populations(self):  # pylint: disable=too-many-locals
        """
        Merge the two populations (old and new) reducing the size by half; update evaluations
        Apply non-conservative approach: prefer newer models if the evaluation equals
        The merged population then becomes the "old" one
        :return: None
        """
        # List evaluations as tuples:
        # - sequence index; 0 == old, 1 == new
        # - index in the population
        # - evaluation value
        old_evaluations = [[0, index, value] for index, value in enumerate(self.population_evaluations)]
        new_evaluations = [[1, index, value] for index, value in enumerate(self.new_population_evaluations)]
        # Sort both lists of evaluations w.r.t. the evaluation, descending
        old_evaluations.sort(key=lambda x: x[2], reverse=True)
        new_evaluations.sort(key=lambda x: x[2], reverse=True)
        # Merge indexes from best to worst, preferring new ones, until population size is reached
        merge_indexes = []
        max_value_pointer_old = 0
        max_value_pointer_new = 0
        while len(merge_indexes) < self.population_size:
            remaining_old_sequence_max_value = old_evaluations[max_value_pointer_old][2]
            remaining_new_sequence_max_value = new_evaluations[max_value_pointer_new][2]
            if remaining_old_sequence_max_value > remaining_new_sequence_max_value:
                merge_indexes.append(old_evaluations[max_value_pointer_old])
                max_value_pointer_old += 1  # Since the list has been sorted, it is enough to move the pointer
            else:
                merge_indexes.append(new_evaluations[max_value_pointer_new])
                max_value_pointer_new += 1
        # Now take only tensor indexes, "unzipping" with regard to old/new
        merge_indexes_old = [item[1] for item in merge_indexes if item[0] == 0]
        merge_indexes_new = [item[1] for item in merge_indexes if item[0] == 1]
        # Use the indexes to extract good models from the populations
        weights_from_old_indexes = self.population_weights[merge_indexes_old]
        biases_from_old_indexes = self.population_biases[merge_indexes_old]
        weights_from_new_indexes = self.new_population_weights[merge_indexes_new]
        biases_from_new_indexes = self.new_population_biases[merge_indexes_new]
        # Concatenate old & new along the 1st (i.e. 0) axis
        self.population_weights = torch.cat((weights_from_old_indexes, weights_from_new_indexes), 0)
        self.population_biases = torch.cat((biases_from_old_indexes, biases_from_new_indexes), 0)
        # Now the merged population becomes the "old" one; updating evaluations
        old_evaluations_values = [self.population_evaluations[index] for index in merge_indexes_old]
        new_evaluations_values = [self.new_population_evaluations[index] for index in merge_indexes_new]
        old_trades_counters = [self.trades_counters[index] for index in merge_indexes_old]
        new_trades_counters = [self.new_trades_counters[index] for index in merge_indexes_new]
        self.population_evaluations = old_evaluations_values + new_evaluations_values
        self.trades_counters = old_trades_counters + new_trades_counters
        self.new_population_weights = None
        self.new_population_biases = None
        self.new_population_evaluations = None
        self.new_trades_counters = None
        # Adaptive mutation probability: tend to have roughly half of models replaced during an epoch
        new_models_percentage = int(len(merge_indexes_new) * 100 / self.population_size)
        if new_models_percentage < 50:
            self.mutation_probability *= 0.95  # Decrease the mutation probability for better stability
            # self.co_probability *= 0.95  # Decrease the cross-over probability for better stability
        else:
            self.mutation_probability *= 1.05  # Increase the mutation probability for better exploration
            self.mutation_probability = min(self.mutation_probability, MAX_MUTATION_PROBABILITY)
            # self.co_probability *= 1.05  # Increase the cross-over probability for better exploration
        logging.info(f'New models taken: {new_models_percentage}%; ' +
                     f'mutation probability: {self.mutation_probability}')

    def _best_model_index(self):
        # There is no argmax for lists in Python(!)
        return max(range(len(self.population_evaluations)), key=lambda i: self.population_evaluations[i])

    def train(self):
        """
        Single epoch
        :return: evaluation of the best model and the number of trades it took
        """
        # Create a new generation and evaluate it
        self.new_generation()
        # Get batch of random sequences from the REST data service
        batch = get_rest_data(self.data_url, self.batch)
        # Conceptual change: both populations, old and new, are evaluated using *the same* batch
        self.population_evaluations, self.trades_counters = self._evaluate_population(
            self.population_weights,
            self.population_biases,
            batch)
        self.new_population_evaluations, self.new_trades_counters = self._evaluate_population(
            self.new_population_weights,
            self.new_population_biases,
            batch)

        # Merge old and new generations, pick the best ones; prefer newer models over old ones
        self._merge_populations()
        best_model_index = self._best_model_index()
        return self.population_evaluations[best_model_index], self.trades_counters[best_model_index]

    def save(self, save_path):
        """
        Saves the population and parameters to the save path. These files are saved:
        - args.txt: command line arguments
        - checkpoint_parameters.json: parameters necessary for the checkpoint
        - weights.pt: tensor of population weights
        - biases.pt: tensor of population biases
        - evaluations.json: models' evaluations
        Population ID is added to file names
        :param save_path: Directory path to save the model
        :return: None
        """
        args_file_name = os.path.join(save_path, self.id + '_args.txt')
        parameters_file_name = os.path.join(save_path, self.id + '_checkpoint_parameters.json')
        weights_file_name = os.path.join(save_path, self.id + '_weights.pt')
        biases_file_name = os.path.join(save_path, self.id + '_biases.pt')
        evaluations_file_name = os.path.join(save_path, self.id + '_evaluations.json')

        checkpoint_parameters = {
            "population_size": self.population_size,
            "model_size": self.model_size,
            "esn_input_size": self.esn_input_size
        }

        with open(args_file_name, 'w') as args_file:
            args_file.write(str(self.args_as_string))
        with open(parameters_file_name, 'w') as parameters_file:
            json.dump(checkpoint_parameters, parameters_file)
        torch.save(self.population_weights, weights_file_name)
        torch.save(self.population_biases, biases_file_name)
        torch.save(self.population_evaluations, evaluations_file_name)


class Model:  # pylint: disable=too-many-instance-attributes
    """
    The class for a single model, for inference / evaluation
    """
    def __init__(self, device, args):
        self.device = device
        self.args = args
        self.max_evaluation_steps = args.max_evaluation_steps
        self.take_profit = args.take_profit
        self.stop_loss = args.stop_loss
        self.id = args.id  # pylint: disable=invalid-name
        # Load ESN input size from checkpoint
        parameters_file_name = os.path.join(args.load_dir, self.id + '_checkpoint_parameters.json')
        with open(parameters_file_name) as parameters_file:
            self.esn_input_size = json.load(parameters_file)["esn_input_size"]
        # Get the best model index from checkpoint
        evaluations_file_name = os.path.join(args.load_dir, self.id + '_evaluations.json')
        population_evaluations = torch.load(evaluations_file_name)
        # There is no argmax in Python(!), hence the workaround
        best_model_index = max(range(len(population_evaluations)), key=lambda i: population_evaluations[i])
        # Get weights and biases of the best model
        weights_file_name = os.path.join(args.load_dir, self.id + "_weights.pt")
        biases_file_name = os.path.join(args.load_dir, self.id + "_biases.pt")
        self.weights = torch.load(weights_file_name)[best_model_index].to(device)
        self.biases = torch.load(biases_file_name)[best_model_index].to(device)

    def infer(self, sequence):
        """
        Run inference on the sequence
        :param sequence: tickers, float, normalized to start at 1.0 value
        :return: trade direction ("buy" or "sell"), trade start point
        """
        internal_state = torch.zeros(self.weights.size()[0], device=self.device)
        relu = torch.nn.ReLU()
        sequence_pointer = 0
        buy_signal = False
        sell_signal = False
        for _ in range(self.args.max_evaluation_steps):
            # Progress the internal state by the model iteration
            for sequence_feed_index in range(self.esn_input_size):
                internal_state[sequence_feed_index] = sequence[sequence_pointer + sequence_feed_index]
            internal_state = torch.einsum("n, mn -> m", internal_state, self.weights)
            internal_state += self.biases
            internal_state = relu(internal_state)

            # The last three state values have a special meaning:
            #     s[-1]: progress input pointer
            #     s[-2]: take long position (buy)
            #     s[-3] : take short position (sell)
            progress_input = (internal_state[-1] >= 1.0).item()
            buy_signal = (internal_state[-2] >= 1.0).item()
            sell_signal = (internal_state[-3] >= 1.0).item()
            # Buy and sell signals mutually cancel out
            if buy_signal and sell_signal:
                buy_signal = False
                sell_signal = False
            if buy_signal or sell_signal:
                break
            if progress_input:
                sequence_pointer += self.esn_input_size
                if sequence_pointer >= len(sequence) - self.esn_input_size:
                    break
        if buy_signal:
            trade = "buy"
        elif sell_signal:
            trade = "sell"
        else:
            trade = "none"
        return trade, sequence_pointer + self.esn_input_size

    def evaluate_sequence(self, sequence):
        """
        Do model inference for the sequence of prices
        :param sequence: array of subsequent prices, normalized to start from 1.0
        :return: gain from $1.0 invested, point of trade start (-1 in case of no trade)
        """
        # Internal state, zero-initialized
        # Note: this is different than in Population, because we have only one model to evaluate
        trade, start_point = self.infer(sequence)
        if trade != "none":
            trade_outcome, trade_executed = calculate_trade_outcome(
                sequence, start_point, trade, self.args.take_profit, self.args.stop_loss)
            return trade_outcome, start_point if trade_executed else -1
        return 0.0, -1

    @staticmethod
    def evaluate_sequence_with_just_buy_strategy(sequence, take_profit, stop_loss):
        """
        Do model inference for the sequence of prices assuming the "buy" decision
        :param sequence: array of subsequent prices, normalized to start from 1.0
        :param take_profit: take profit value (1.05 == 5%)
        :param stop_loss: stop loss value (0.95 == -5%)
        :return: gain from $1.0 invested
        """
        trade_outcome, _ = calculate_trade_outcome(
            sequence, 0, "long", take_profit, stop_loss)
        return trade_outcome, 0
