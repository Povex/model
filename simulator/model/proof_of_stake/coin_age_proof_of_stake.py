import logging
import random

from simulator.model.proof_of_stake.proof_of_stake import PoS
from simulator.model.initial_distributions import Distributions
from simulator.model.agents.node_agent_v2 import NodeAgentV2


class CoinAgePoS(PoS):

    def __init__(self):
        super().__init__()
        self.create_agents()

    def create_agents(self):
        stakes = Distributions.generate_distribution(self.conf.initial_distribution,
                                                     self.conf.n_agents,
                                                     self.conf.initial_stake_volume)
        for i in range(self.conf.n_agents):
            a = NodeAgentV2(i)
            a.add_stake(stakes[i], 0)
            self.agents.add(a)
            self.stake_volume += stakes[i]
        self.agents = frozenset(self.agents)
        self.update_history()

    def select_validator(self):
        agents = list(self.agents.difference(self.agents_stop_epochs))
        return random.choices(agents,
                              weights=[a.get_stake_weight(
                                  self.epoch,
                                  self.conf.min_coin_age,
                                  self.conf.max_coin_age
                              ) for a in agents])[0]

    def consensus(self):
        try:
            validator = self.select_validator()
        except IndexError:
            return

        p = self.conf.malicious_node_probability
        validator.is_malicious = random.choices((True, False), weights=(p, 1 - p))[0]

        if validator.is_malicious:
            logging.info(f"Malicious node found with id {validator.unique_id} at epoch {self.epoch}")
            stake_penalty = validator.stake * self.conf.percent_stake_penalty
            validator.stake.remove_stake(stake_penalty)
            self.stake_volume -= stake_penalty
            validator.stop_epochs += self.conf.stop_epoch_after_malicious
            validator.is_malicious = False

        else:
            logging.debug(f"Validator found with id {validator.unique_id} at epoch {self.epoch}")
            validator.stop_epochs += self.conf.stop_epoch_after_validator
            old_stake = validator.get_stake()
            validator.add_stake(self.conf.block_reward, self.epoch)
            current_stake = validator.get_stake()
            if current_stake > self.conf.stake_limit:
                validator.remove_stake(current_stake - self.conf.stake_limit)
            self.stake_volume += (validator.get_stake() - old_stake)
            # Reset the age of the coins
            for coins in validator.stake:
                coins.age = self.epoch

        if validator.stop_epochs > 0:
            self.agents_stop_epochs.add(validator)

