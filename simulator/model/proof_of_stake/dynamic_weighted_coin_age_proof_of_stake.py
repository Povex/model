import logging
import random

from simulator import CoinAgePoS
from simulator.model.statistics.statistics import gini_irc


class DynamicWeightedCoinAge(CoinAgePoS):

    def __init__(self):
        super().__init__()

    def weighted_coin_age(self):
        agents = list(self.agents.difference(self.agents_stop_epochs))
        return random.choices(agents,
                              weights=[a.get_stake_weight(
                                  self.epoch,
                                  self.conf.min_coin_age,
                                  self.conf.max_coin_age
                              ) for a in agents])[0]

    def inverse_weighted_coin_age(self):
        agents = list(self.agents.difference(self.agents_stop_epochs))
        weight_volume = sum([a.get_stake_weight(self.epoch, self.conf.min_coin_age, self.conf.max_coin_age) for a in self.agents])
        inverse_weights = [(weight_volume - a.get_stake_weight(self.epoch, self.conf.min_coin_age, self.conf.max_coin_age)) / weight_volume for a in agents]
        return random.choices(agents, weights=inverse_weights)[0]

    def select_validator(self):
        gini = gini_irc([a.get_stake() for a in self.agents])
        if gini > self.conf.gini_threshold:
            return self.inverse_weighted_coin_age()
        return self.weighted_coin_age()
