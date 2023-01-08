import random

from simulator.model.proof_of_stake.proof_of_stake import PoS
from simulator.model.statistics.statistics import gini_irc


class DynamicWeighted(PoS):
    """
    Il sistema dinamico permette di modificare il modello al variare di certe condizioni, per esempio al raggiungimento
    di una soglia nell'indice di Gini.
     """

    def __init__(self):
        super().__init__()
        self.create_agents()

    def weighted(self):
        agents = list(self.agents.difference(self.agents_stop_epochs))
        return random.choices(agents, weights=[a.stake for a in agents])[0]

    def inverse_weighted(self):
        agents = list(self.agents.difference(self.agents_stop_epochs))
        inverse_stakes = [(self.stake_volume - a.stake) / self.stake_volume for a in agents]
        return random.choices(agents, weights=inverse_stakes)[0]

    def select_validator(self):
        gini = gini_irc([a.stake for a in self.agents])  # TODO: use np version
        if gini > self.conf.gini_threshold:
            return self.inverse_weighted()
        return self.weighted()
