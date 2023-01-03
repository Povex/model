from dataclasses import dataclass
import json
from typing import List


class Singleton(type):
    _instances: dict = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


@dataclass
class ModelConfig(metaclass=Singleton):
    n_agents: int
    n_epochs: int
    pos_type: str
    initial_distribution: str
    gini_initial_distribution: float
    custom_distribution: List[float]
    initial_stake_volume: float
    block_reward: float
    gini_threshold: float
    log_weighted_base: float
    log_shift_factor: float
    malicious_node_probability: float
    percent_stake_penalty: float
    stop_epoch_after_malicious: int
    stop_epoch_after_validator: int
    stake_limit: float
    min_coin_age: int
    max_coin_age: int
    early_withdrawing_penalty: float
    weighted_reward: float

    def __init__(self,  model_config=None, path='model_config.json'):
        if model_config is not None:
            self.__dict__ = model_config
            return
        with open(path, 'r') as f:
            self.__dict__ = json.load(f)

    def set_model_config(self, model_config):
        self.__dict__ = model_config

