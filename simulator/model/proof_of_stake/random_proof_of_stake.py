import random

from simulator.model.proof_of_stake.proof_of_stake import PoS


class RandomPoS(PoS):
    """
        Random proof of stake is a class that describe a proof_of_stake consensus in which the agent
        responsible to mine the next block is selected randomly.
    """
    def __init__(self):
        super().__init__()
        self.create_agents()

    def select_validator(self):
        agents = list(self.agents.difference(self.agents_stop_epochs))
        return random.choice(agents)

