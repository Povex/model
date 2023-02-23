from typing import List

import numpy as np
import pandas as pd

from simulator.model.statistics.statistics import gini_concentration_index


# TODO: Check if ['mean'] is a good choice or the std is important too
class Metrics:
    gini_stakes_diff = 'gini_stakes_diff'
    gini_rewards_diff = 'gini_rewards_diff'
    slope_gini_stakes = 'slope_gini_stakes'
    slope_gini_rewards = 'slope_gini_rewards'

    def __init__(self, history: pd.DataFrame):
        super().__init__()
        self.history = history
        self.first_epoch = 0
        self.last_epoch = self.history["epoch"].max()

    def get_gini_rewards_diff(self):
        stakes_first_epoch = self.history.query(f"epoch == {self.first_epoch}").drop(columns=['epoch'])
        stakes_last_epoch = self.history.query(f"epoch == {self.last_epoch}").drop(columns=['epoch'])
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
        """
        If the difference is positive then the inequality increased
        If the difference is 0 then the optimum is reached
        If the difference is negative then the inequality decreased
        :return: Difference between last epoch Gini and first epoch Gini
        """
        gini_first_epoch = self.history.query(f'epoch == {self.first_epoch}') \
            .drop(columns=['id', 'epoch']) \
            .groupby(['simulation'])["stake"] \
            .apply(gini_concentration_index) \
            .reset_index()

        gini_last_epoch = self.history.query(f'epoch == {self.last_epoch}') \
            .drop(columns=['id', 'epoch']) \
            .groupby(['simulation'])["stake"] \
            .apply(gini_concentration_index) \
            .reset_index()

        gini_diff_mean = gini_first_epoch.merge(gini_last_epoch, on='simulation', suffixes=('', '_')) \
            .eval("stake = stake_ - stake") \
            .drop(columns=['stake_'])["stake"] \
            .mean()  # TODO: Check if mean is a good choice or the std is important too

        return gini_diff_mean

    def get_slope_gini_stakes(self):
        values = list(range(0, self.last_epoch + 1, int(self.last_epoch * .1)))
        history = self.history.query("epoch in @values")
        initial_gini = gini_concentration_index(
            self.history.query(f'epoch == {self.first_epoch} and simulation == 0')["stake"])
        slope_mean = history.groupby(['simulation', 'epoch'])['stake'] \
            .apply(gini_concentration_index) \
            .groupby(['epoch']) \
            .mean() \
            .reset_index() \
            .assign(initial_gini=initial_gini) \
            .eval('stake = (stake - initial_gini) / epoch') \
            .replace([np.inf, -np.inf], 0)['stake'] \
            .mean()
        return slope_mean

    def get_slope_gini_rewards(self):
        values = list(range(0, self.last_epoch + 1, int(self.last_epoch * .1)))
        history = self.history.query("epoch in @values")
        initial_gini = gini_concentration_index(
            self.history.query(f'epoch == {self.first_epoch} and simulation == 0')["stake"])
        stakes_first_epoch = self.history.query(f"epoch == {self.first_epoch}").drop(columns=['epoch'])
        slope_mean = history.merge(stakes_first_epoch, on=['simulation', 'id'], suffixes=('', '_')) \
            .eval("stake = stake - stake_") \
            .drop(columns=['stake_']) \
            .groupby(['simulation', 'epoch'])['stake'] \
            .apply(gini_concentration_index) \
            .groupby(['epoch']) \
            .mean() \
            .reset_index() \
            .assign(initial_gini=initial_gini) \
            .eval('stake = (stake - initial_gini) / epoch') \
            .replace([np.inf, -np.inf], 0)['stake'] \
            .mean()
        return slope_mean

    def scores(self, metrics: List[str]) -> dict:
        scores = {}
        if Metrics.gini_stakes_diff in metrics:
            scores[Metrics.gini_stakes_diff] = self.get_gini_stakes_diff()
        if Metrics.gini_rewards_diff in metrics:
            scores[Metrics.gini_rewards_diff] = self.get_gini_rewards_diff()
        if Metrics.slope_gini_stakes in metrics:
            scores[Metrics.slope_gini_stakes] = self.get_slope_gini_stakes()
        if Metrics.slope_gini_rewards in metrics:
            scores[Metrics.slope_gini_rewards] = self.get_slope_gini_rewards()
        return scores
