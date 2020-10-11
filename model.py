import os
import random

import torch


def _coin_toss():
    if random.random() > 0.5:
        return True
    return False


class Population:
    """
    Model candidates
    """
    def __init__(self, device, population_size, model_size, data_stream):
        self.device = device
        self.population_size = population_size
        self.model_size = model_size
        # Random initialization, uniform distribution from [0, 1)
        self.population_weights = torch.rand([population_size, model_size, model_size], device=device)
        self.population_biases = torch.rand([population_size, model_size], device=device)
        # Change distribution from [0, 1) to [-1, 1)
        self.population_weights = self.population_weights * 2 - 1
        self.population_biases = self.population_biases * 2 - 1
        self.new_population_weights = None
        self.new_population_biases = None
        self.data_stream = data_stream
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

    def _mutate_new_population(self, mutation_probability=0.001):
        """
        Mutate new population randomly, by increasing or decreasing by a small value
        Note: this code does not need to be very efficient, it will not be run frequently
        :param mutation_probability: probability of a single value mutation
        :return: None
        """
        for i in range(self.population_size):
            for j in range(self.model_size):
                for k in range(self.model_size):
                    if random.random() < mutation_probability:
                        if _coin_toss():
                            if _coin_toss():
                                # TODO consider other mutation methods
                                self.new_population_weights[i, j, k] -= 0.0001
                            else:
                                self.new_population_weights[i, j, k] += 0.0001
        for i in range(self.population_size):
            for j in range(self.model_size):
                if random.random() < mutation_probability:
                    if _coin_toss():
                        if _coin_toss():
                            self.new_population_biases[i, j] -= 0.0001
                        else:
                            self.new_population_biases[i, j] += 0.0001

    def _crossing_over_in_new_population(self, co_probability=0.4):
        """
        Cross-over models randomly
        :param co_probability: crossing-over probability
        :return:
        """
        model_size_squared = self.model_size * self.model_size
        # Create a reshaped view of new population weights; values are shared, so we can do in situ modifications
        weights_reshaped = self.new_population_weights.view(self.population_size, model_size_squared)
        # Sample all pairs from the new population randomly
        crossing_over_already_done = [False] * self.population_size  # Flags to note if c.o. happened
        while True:
            remaining_population = [index for index in range(self.population_size)
                                    if not crossing_over_already_done[index]]
            if len(remaining_population) < 2:
                break
            first_index, second_index = random.sample(remaining_population, 2)
            crossing_over_already_done[first_index] = True
            crossing_over_already_done[second_index] = True
            if random.random() < co_probability:  # Actual crossing over happens here
                # Weights:
                # Select the split point; at least one element from either side needs to be taken
                co_split_index = random.choice(range(1, model_size_squared - 1))
                # Swap sequences
                saved_left_sequence = self.new_population_weights[first_index][:co_split_index]
                saved_right_sequence = self.new_population_weights[second_index][co_split_index:]
                self.new_population_weights[first_index][:co_split_index] =\
                    self.new_population_weights[second_index][:co_split_index]
                self.new_population_weights[second_index][co_split_index:] =\
                    self.new_population_weights[first_index][co_split_index:]
                self.new_population_weights[second_index][:co_split_index] = saved_left_sequence
                self.new_population_weights[first_index][co_split_index:] = saved_right_sequence
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

    def _evaluate_population(self, weights, biases, data_stream):
        """
        The main evaluation method; takes a population (as weights & biases), the data stream,
        and returns trade values for models in the population
        :param weights: array of population models' weights
        :param biases: array of biases
        :param data_stream: the stream of data batches, as lists of numpy arrays
        :return: population-sized list of trade results
        """
        return [random.random() for _ in range(weights.shape[0])]  # TODO implement

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
        old_evaluations_copy = old_evaluations.copy()  # Full lists will be useful further
        new_evaluations_copy = new_evaluations.copy()
        while len(merge_indexes) < self.population_size:
            remaining_old_sequence_max_value = old_evaluations_copy[0][2]
            remaining_new_sequence_max_value = new_evaluations_copy[0][2]
            if remaining_old_sequence_max_value > remaining_new_sequence_max_value:
                merge_indexes.append(old_evaluations_copy[0])  # Take the head of the list
                old_evaluations_copy = old_evaluations_copy[1:]  # Discard after use
            else:
                merge_indexes.append(new_evaluations_copy[0])
                new_evaluations_copy = new_evaluations_copy[1:]
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
        # Take the best result so far, to return it at the end
        best_result_so_far = max(self.population_evaluations + self.new_population_evaluations)
        # Merge old and new generations, pick the best ones; prefer newer models over old ones
        self._merge_populations()
        return best_result_so_far

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
