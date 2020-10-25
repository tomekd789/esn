"""
Model emulation by a random number generator, to observe maximum possible gains
"""
from math import exp, log
import random

BATCH = 2000
WEEKS_PER_SEQUENCE = 2
POPULATION = 50
EPOCHS = 100000

max_gain = 0.0
for i in range(EPOCHS):
    for j in range(POPULATION):
        best_result_so_far = sum([random.choice([-1, 1]) for _ in range(BATCH)]) * 0.05
        trade_duration_weeks = BATCH * WEEKS_PER_SEQUENCE
        trade_duration_years = trade_duration_weeks / 52
        best_result_per_year = best_result_so_far / trade_duration_years
        yearly_gain_percent = best_result_per_year * 100  # Cash starts at $1.00
        if yearly_gain_percent > max_gain:
            max_gain = yearly_gain_percent
    if i in [0, 9, 99, 999, 9999, 99999]:
        print(f'Epoch: {i + 1}; maximum yearly gain: {max_gain:.2f}%')
