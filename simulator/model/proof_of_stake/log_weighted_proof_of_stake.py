import math
import random

from simulator.model.proof_of_stake.proof_of_stake import PoS

class LogWeightedPoS(PoS):

    def __init__(self):
        super().__init__()
        self.create_agents()

    def select_validator(self):
        agents = list(self.agents.difference(self.agents_stop_epochs))
        agents_volume = sum([a.stake for a in agents])
        weights = [a.stake / agents_volume for a in agents]
        log_weights = [math.log(self.conf.log_shift_factor + w, self.conf.log_weighted_base) for w in weights]
        return random.choices(agents, weights=log_weights)[0]
