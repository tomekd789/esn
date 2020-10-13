import os
import random

import torch

MUTATION_FUNCTIONS = [
    lambda x: x / 2,
    lambda x: x * 2,
    lambda x: x - 0.5,
    lambda x: x + 0.5
]


def _get_mutation_function(mutation_probability):
    if random.random() < mutation_probability:
        return lambda x: x
    else:
        return random.choice(MUTATION_FUNCTIONS)


class Population:
    """
    Model candidates
    """
    def __init__(self, device, data, args):
        self.device = device
        self.population_size = args.population
        self.model_size = args.model_size
        self.mutation_probability = args.mutation_probability
        self.co_probability = args.co_probability
        self.max_evaluation_steps = args.max_evaluation_steps
        self.take_profit = args.take_profit
        self.stop_loss = args.stop_loss
        # Random initialization, uniform distribution from [0, 1)
        # self.population_weights = torch.rand([population_size, model_size, model_size], device=device)
        # self.population_biases = torch.rand([population_size, model_size], device=device)
        # Change distribution from [0, 1) to [-1, 1)
        # self.population_weights = self.population_weights * 2 - 1
        # self.population_biases = self.population_biases * 2 - 1
        # Zero initialization
        self.population_weights = torch.zeros([self.population_size, self.model_size, self.model_size], device=device)
        self.population_biases = torch.zeros([self.population_size, self.model_size], device=device)
        self.new_population_weights = None
        self.new_population_biases = None
        self.data_stream = data
        # Note that this is an extra population evaluation, it takes additional time
        # In case of zero initialization we might skip it, but this is risky if initialization method changes later
        self.population_evaluations = self._evaluate_population(
            self.population_weights,
            self.population_biases,
            self.data_stream)
        self.new_population_evaluations = None

    def new_generation(self):
        """
        Takes a copy of the existing population and applies mutations and crossing-over to it
        :return: None
        """
        self._copy_population()
        self._mutate_new_population()
        self._crossing_over_in_new_population()
        self.new_population_evaluations = self._evaluate_population(
            self.new_population_weights,
            self.new_population_biases,
            self.data_stream)

    def _copy_population(self):
        """
        Create a copy of the existing population; it will be randomly modified and evaluated
        :return:
        """
        self.new_population_weights = self.population_weights.clone()
        self.new_population_biases = self.population_biases.clone()

    def _mutate_new_population(self):
        """
        Mutate new population randomly, by increasing or decreasing by a small value
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

    def _calculate_trade_outcome(self, sequence, trade_start_pointer, trade_type):
        trade_start_price = sequence[trade_start_pointer]
        # We start this calculation with $1.00 and buy at the trade start price
        # To simulate continuous trading, we multiply the outcomes from single sequences
        purchased_stocks = 1.0 / trade_start_price
        sequence_index = trade_start_pointer + 1
        while sequence_index < len(sequence) - 1:
            # Check for Take Profit
            if sequence[sequence_index] >= trade_start_price * self.take_profit:
                break
            # Check for Stop Loss
            if sequence[sequence_index] <= trade_start_price * self.stop_loss:
                break
            sequence_index += 1
        trade_close_price = sequence[sequence_index].astype(float)
        result = purchased_stocks * trade_close_price
        if trade_type == "short":
            # I calculate the gain if the trade type would be long,
            # and subtract it from the initial wallet state (I prefer to write it verbatim)
            gain = result - 1.0
            result = 1.0 - gain
        return result

    def _evaluate_sequence(self, weights, biases, sequence):
        """
        Evaluate a single sequence
        :param weights: population models' weights
        :param biases: population models' biases
        :param sequence: single trade sequence as a python list
        :return: population-sized tensor of trade results
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
        # Perform RNN steps with all models in parallel (leverage the CUDA SIMD architecture)
        relu = torch.nn.ReLU()
        for _ in range(self.max_evaluation_steps):
            # For each internal state assign its 0-th element with the sequence value according to sequence pointers
            pointers = sequence_pointers.tolist()
            for model_index, pointer in enumerate(pointers):
                internal_states[model_index][0] = sequence[int(pointer)].astype(float)

            """                    THE CORE OPERATION.  Efficiency is much in demand here                   """
            # Scalar multiply internal states by corresponding models, add biases
            internal_states = torch.einsum("bn, bmn -> bm", internal_states, weights)
            internal_states += biases
            internal_states = relu(internal_states)
            """                                                                                             """

            # The last three state values have a special meaning:
            #     s[-3]: progress input pointer
            #     s[-2]: take long position (buy)
            #     s[-1] : take short position (sell)
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
            sequence_pointers += step_pointers_forward
            # Terminate the calculations if all trade signals have been switched on
            if is_trade.all():
                break
        # Calculate trades results, model by model; initial wallet state is 1.0
        trades_results = torch.ones(self.population_size, device=self.device)
        for model_index in range(self.population_size):
            # If no trade was signalled, keep the existing value, i.e. 1.0
            if not is_trade[model_index]:
                continue
            trade_start_pointer = int(trade_start_pointers[model_index].item())
            trade_type = "long" if take_long_position[model_index] else "short"
            trades_results[model_index] =\
                self._calculate_trade_outcome(sequence, trade_start_pointer, trade_type)
        return trades_results

    def _evaluate_population(self, weights, biases, data_stream):
        """
        The main evaluation method; takes a population (as weights & biases), the data stream,
        and returns trade values for models in the population
        :param weights: array of population models' weights
        :param biases: array of biases
        :param data_stream: the stream of data batches, as lists of numpy arrays
        :return: population-sized list of trade results
        """
        batch = data_stream.__next__()
        # Note that batch elements are processed sequentially; the name might be counterintuitive for DL practitioners
        # The calculations are rather batched along the population, each model being evaluated independently
        accumulated_evaluation = torch.ones(self.population_size, device=self.device)
        #a = 0  # TODO remove after debug
        for sequence in batch:
            #print(a); a += 1  # TODO remove after debug
            accumulated_evaluation *= self._evaluate_sequence(weights, biases, sequence)
        return accumulated_evaluation.tolist()

    def _merge_populations(self):
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
        merge_indexes_old = [tuple[1] for tuple in merge_indexes if tuple[0] == 0]
        merge_indexes_new = [tuple[1] for tuple in merge_indexes if tuple[0] == 1]
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
        self.population_evaluations = old_evaluations_values + new_evaluations_values
        self.new_population_weights = None
        self.new_population_biases = None
        self.new_population_evaluations = None

    def train(self):
        # Create a new generation and evaluate it
        self.new_generation()
        # Merge old and new generations, pick the best ones; prefer newer models over old ones
        self._merge_populations()
        return max(self.population_evaluations)

    def save(self, save_path):
        """
        Saves the best model (i.e. indexed as 0) to the save path, with names 'weights.pt' and 'biases.pt'
        :param save_path: Directory path to save the model
        :return: None
        """
        # There is no argmax for lists in Python(!)
        best_model_index = max(range(len(self.population_evaluations)), key=lambda i: self.population_evaluations[i])
        torch.save(self.population_weights[best_model_index], os.path.join(save_path, 'weights.pt'))
        torch.save(self.population_biases[best_model_index], os.path.join(save_path, 'biases.pt'))
