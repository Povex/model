import logging
import random
from typing import Set

import pandas as pd

from simulator.model.initial_distributions import Distributions
from simulator.model.model_config import ModelConfig
from simulator.model.agents.node_agent_v1 import NodeAgentV1

from tqdm import tqdm


class PoS:

    def __init__(self):
        self.conf = ModelConfig()
        self.agents: Set[NodeAgentV1] = set()
        self.stake_volume = 0
        self.epoch = 0
        self.history = []
        self.agents_stop_epochs = set()

    def step(self):
        self.epoch += 1
        self.consensus()
        self.update_stop_epochs()
        self.update_history()

    def run(self):
        for _ in tqdm(range(self.conf.n_epochs)):
            self.step()
        self.history = pd.DataFrame.from_dict(self.history)

    def update_history(self):
        self.history.extend([{'epoch': self.epoch, 'id': a.unique_id, 'stake': a.get_stake()} for a in self.agents])

    def update_stop_epochs(self):
        free_agents = []
        for agent in self.agents_stop_epochs:
            agent.stop_epochs -= 1
            if agent.stop_epochs == 0:
                free_agents.append(agent)
        for agent in free_agents:
            self.agents_stop_epochs.remove(agent)

    def create_agents(self):
        stakes = Distributions.generate_distribution(self.conf.initial_distribution,
                                                     self.conf.n_agents,
                                                     self.conf.initial_stake_volume)
        for i in range(self.conf.n_agents):
            self.agents.add(NodeAgentV1(i, stakes[i]))
            self.stake_volume += stakes[i]
        self.agents = frozenset(self.agents)
        self.update_history()

    def select_validator(self):
        pass

    def get_block_reward(self):
        if self.conf.reward_type == "constant":
            return self.conf.total_rewards/self.conf.n_epochs
        if self.conf.reward_type == "geometric":
            return (1 + self.conf.total_rewards) ** (self.epoch / self.conf.n_epochs) \
                - (1 + self.conf.total_rewards) ** ((self.epoch - 1) / self.conf.n_epochs)
        raise Exception("Reward function must be 'constant' or 'geometric'.")

    def consensus(self):
        try:
            validator = self.select_validator()
        except IndexError:
            logging.info(f"Validator not found")
            return

        p = self.conf.malicious_node_probability
        validator.is_malicious = random.choices((True, False), weights=(p, 1 - p))[0]

        if validator.is_malicious:
            logging.info(f"Malicious node found with id {validator.unique_id} at epoch {self.epoch}")
            stake_penalty = validator.stake * self.conf.percent_stake_penalty
            validator.stake -= stake_penalty
            self.conf.initial_stake_volume -= stake_penalty

            validator.stop_epochs += self.conf.stop_epoch_after_malicious
            validator.is_malicious = False

        else:
            logging.debug(f"Validator found with id {validator.unique_id} at epoch {self.epoch}")
            validator.stop_epochs += self.conf.stop_epoch_after_validator
            old_stake = validator.stake
            validator.stake += self.get_block_reward()
            if validator.stake > self.conf.stake_limit:
                validator.stake = self.conf.stake_limit
            self.stake_volume += (validator.stake - old_stake)

        if validator.stop_epochs > 0:
            self.agents_stop_epochs.add(validator)
