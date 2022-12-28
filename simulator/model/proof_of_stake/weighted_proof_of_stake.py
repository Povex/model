import random

from simulator.model.proof_of_stake.proof_of_stake import PoS


class WeightedPoS(PoS):

    def __init__(self):
        super().__init__()
        self.create_agents()

    def select_validator(self):
        agents = list(self.agents.difference(self.agents_stop_epochs))
        return random.choices(agents, weights=[a.stake for a in agents])[0]
