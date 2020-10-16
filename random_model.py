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
        log_best_result_so_far = (sum([random.choice([0, 1]) for _ in range(BATCH)]) - BATCH / 2) * log (1.05)
        trade_duration_weeks = BATCH * WEEKS_PER_SEQUENCE
        trade_duration_years = trade_duration_weeks / 52
        log_best_result_per_year = log_best_result_so_far / trade_duration_years
        wallet_after_a_year = exp(log_best_result_per_year)
        yearly_gain_percent = (wallet_after_a_year - 1.0) * 100  # Wallet value starts at $1.00
        if yearly_gain_percent > max_gain:
            max_gain = yearly_gain_percent
    if i in [0, 9, 99, 999, 9999, 99999]:
        print(f'Epoch: {i + 1}; maximum yearly gain: {max_gain:.2f}%')
