from typing import List

import pandas as pd

from simulator.model.statistics.statistics import gini_concentration_index


# TODO: Check if ['mean'] is a good choice or the std is important too
class Metrics:
    gini_stakes_diff = 'gini_stakes_diff'
    gini_rewards_diff = 'gini_rewards_diff'

    def __init__(self, history: pd.DataFrame):
        super().__init__()
        self.history = history

    def get_gini_rewards_diff(self):
        first_epoch = 0
        last_epoch = self.history["epoch"].max()

        stakes_first_epoch = self.history.query(f"epoch == {first_epoch}").drop(columns=['epoch'])
        stakes_last_epoch = self.history.query(f"epoch == {last_epoch}").drop(columns=['epoch'])
        rewards_last_epoch = stakes_last_epoch.merge(stakes_first_epoch, on=['simulation', 'id'], suffixes=('', '_')) \
            .eval("stake = stake - stake_") \
            .drop(columns=['stake_', 'id']) \
            .groupby(['simulation'])['stake'] \
            .apply(gini_concentration_index) \
            .reset_index() \
            .drop(columns=['simulation']) \
            .mean()  # TODO: Check if mean is a good choice or the std is important too

        return rewards_last_epoch[0]

    def get_gini_stakes_diff(self):
        first_epoch = 0
        last_epoch = self.history["epoch"].max()

        gini_first_epoch = self.history.query(f'epoch == {first_epoch}') \
            .drop(columns=['id', 'epoch']) \
            .groupby(['simulation'])["stake"] \
            .apply(gini_concentration_index) \
            .reset_index()

        gini_last_epoch = self.history.query(f'epoch == {last_epoch}') \
            .drop(columns=['id', 'epoch']) \
            .groupby(['simulation'])["stake"] \
            .apply(gini_concentration_index) \
            .reset_index()

        gini_diff_mean = gini_first_epoch.merge(gini_last_epoch, on='simulation', suffixes=('', '_')) \
            .eval("stake = stake_ - stake") \
            .drop(columns=['stake_'])["stake"] \
            .mean()  # TODO: Check if mean is a good choice or the std is important too

        return gini_diff_mean

    def scores(self, metrics: List[str]) -> dict:
        scores = {}
        if Metrics.gini_stakes_diff in metrics:
            scores[Metrics.gini_stakes_diff] = self.get_gini_stakes_diff()
        if Metrics.gini_rewards_diff in metrics:
            scores[Metrics.gini_rewards_diff] = self.get_gini_rewards_diff()
        return scores
