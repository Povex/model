import random

from simulator.model.proof_of_stake.proof_of_stake import PoS


class InverseWeightedPoS(PoS):

    def __init__(self):
        super().__init__()
        self.create_agents()

    def select_validator(self):
        """
        The inverse stake corresponds to the inverse probability distribution
        """
        agents = list(self.agents.difference(self.agents_stop_epochs))
        inverse_stakes = [(self.stake_volume - a.stake) / self.stake_volume for a in agents]
        return random.choices(agents, weights=inverse_stakes)[0]
