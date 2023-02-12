import logging
import math
import random

from simulator.model.proof_of_stake.proof_of_stake import PoS


class CoinAgePoS(PoS):

    def __init__(self):
        super().__init__()
        self.create_agents()

    def update_coin_ages(self):
        for a in self.agents:
            a.coin_age += 1

    def step(self):
        self.epoch += 1
        self.update_stop_epochs()
        self.update_coin_ages()
        self.consensus()
        self.update_history()

    def select_validator(self):
        return random.choices(list(self.agents), weights=[a.get_coin_age_weight() for a in self.agents])[0]

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
            validator.coin_age -= math.ceil(self.conf.coin_age_reduction_factor * validator.coin_age)
            old_stake = validator.stake
            validator.stake += self.get_block_reward()
            if validator.stake > self.conf.stake_limit:
                validator.stake = self.conf.stake_limit
            self.stake_volume += (validator.stake - old_stake)

        if validator.stop_epochs > 0:
            self.agents_stop_epochs.add(validator)
