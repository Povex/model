import math
from typing import List

import simulator as sim
from metrics import Metrics



def get_default_model_config():
    return {
        "n_agents": 10,
        "n_epochs": 1000,
        "initial_stake_volume": 1000.0,
        "block_reward": 1.0,

        "stop_epoch_after_validator": 0,
        "stake_limit": 999999999999999.0,

        "malicious_node_probability": 0.0,
        "stop_epoch_after_malicious": 5,
        "percent_stake_penalty": 0.2,

        "pos_type": "random",
        "initial_distribution": "polynomial",
        "gini_initial_distribution": 0.3,
        "custom_distribution": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "gini_threshold": 0.8,
        "log_weighted_base": 2,
        "log_shift_factor": 1,
        "dynamic_weighted_theta": 0.5,

        "min_coin_age": 0,
        "max_coin_age": 999999999999999,

        "early_withdrawing_penalty": 0.2,
        "weighted_reward": 1.0
    }


class Experiments:

    def __init__(self, metrics: List[str], n_simulations=4):
        super().__init__()
        self.metrics = metrics
        self.current_model_config = get_default_model_config()
        sim.ModelConfig(self.current_model_config)
        self.scores = None
        self.history = None
        self.optimums = self.init_optimums()
        self.n_simulations = n_simulations

    def init_optimums(self):
        return {metric: {'min': {'value': math.inf,
                                 'history': None,
                                 'model_config': None},
                         'max': {'value': -math.inf,
                                 'history': None,
                                 'model_config': None}
                         } for metric in self.metrics}

    def update_optimums(self):
        for metric in self.metrics:
            if self.scores[metric] < self.optimums[metric]['min']['value']:
                self.optimums[metric]['min']['value'] = self.scores[metric]
                self.optimums[metric]['min']['history'] = self.history
                self.optimums[metric]['min']['model_config'] = dict(sim.ModelConfig().__dict__)
            if self.scores[metric] > self.optimums[metric]['max']['value']:
                self.optimums[metric]['max']['value'] = self.scores[metric]
                self.optimums[metric]['max']['history'] = self.history
                self.optimums[metric]['max']['model_config'] = dict(sim.ModelConfig().__dict__)

    def run_experiment(self):
        sim.ModelConfig().set_model_config(self.current_model_config)
        self.history = sim.ModelRunner.run(n_simulations=self.n_simulations)
        self.scores = Metrics(self.history).scores(self.metrics)
        self.update_optimums()

    def run(self):
        initial_stake_volume = self.current_model_config['initial_stake_volume']
        for stop_epoch_after_validator in (0, 2, 5):
            self.current_model_config['stop_epoch_after_validator'] = stop_epoch_after_validator
            for stake_limit in (initial_stake_volume, math.inf):
                self.current_model_config['stake_limit'] = stake_limit
                for pos_type in ('random', 'weighted', 'inverse_weighted', 'log_weighted', 'dynamic_weighted'):
                    self.current_model_config['pos_type'] = pos_type
                    for initial_distribution in ('gini',):
                        self.current_model_config['initial_distribution'] = initial_distribution
                        for gini_initial_distribution in (.0, .2, .4, .6, .8, .99999):
                            self.current_model_config['gini_initial_distribution'] = gini_initial_distribution
                            self.run_experiment()
                    for initial_distribution in ('linear',):
                        self.current_model_config['initial_distribution'] = initial_distribution
                        self.run_experiment()
        return self.optimums


