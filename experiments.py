import math
import sys

import simulator as sim
import pickle

import os
import shutil
from pathlib import Path

from data_visualization import *
from experiments_visualization import ExperimentsVisualization
from metrics import Metrics
import json

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def get_default_model_config():
    return {
        "n_agents": 10,
        "n_epochs": 10000,
        "initial_stake_volume": 1000.0,
        "total_rewards": 10000,
        'reward_type': 'constant',

        "stop_epoch_after_validator": 0,
        "stake_limit": 999999999999999.0,

        "malicious_node_probability": 0.0,
        "stop_epoch_after_malicious": 5,
        "percent_stake_penalty": 0.2,

        "pos_type": "coin_age",
        "initial_distribution": "polynomial",
        "gini_initial_distribution": 0.3,
        "custom_distribution": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "gini_threshold": 0.8,
        "log_weighted_base": 2,
        "log_shift_factor": 1,
        "dynamic_weighted_theta": 0.5,
        "coin_age_reduction_factor": 0,

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
        self.n_simulations = n_simulations
        self.optimums = self.init_optimums()
        self.all_scores = self.init_all_scores()

    def init_optimums(self):
        return {metric: {'min': {'value': math.inf,
                                 'history': None,
                                 'model_config': None},
                         'max': {'value': -math.inf,
                                 'history': None,
                                 'model_config': None}
                         } for metric in self.metrics}

    def init_all_scores(self):
        return {metric: [] for metric in self.metrics}

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

    def update_all_scores(self):
        for metric in self.metrics:
            self.all_scores[metric].append(
                {
                    'model_config': dict(sim.ModelConfig().__dict__),
                    'value': self.scores[metric]
                }
            )

    def run_experiment(self):
        sim.ModelConfig().set_model_config(self.current_model_config)
        self.history = sim.ModelRunner.run(n_simulations=self.n_simulations)
        self.scores = Metrics(self.history).scores(self.metrics)
        # self.update_optimums()
        self.update_all_scores()

    def run(self):
        initial_stake_volume = self.current_model_config['initial_stake_volume']
        for stop_epoch_after_validator in (0,):  # 2, 5):
            self.current_model_config['stop_epoch_after_validator'] = stop_epoch_after_validator
            for stake_limit in (initial_stake_volume, math.inf):
                self.current_model_config['stake_limit'] = stake_limit
                for pos_type in ('random', 'weighted', 'coin_age', 'dynamic_weighted'):
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

    def run_random(self):
        self.current_model_config['pos_type'] = 'random'
        self.current_model_config['initial_distribution'] = 'gini'
        initial_stake_volume = self.current_model_config['initial_stake_volume']
        for stop_epoch_after_validator in (0, 2, 5):
            self.current_model_config['stop_epoch_after_validator'] = stop_epoch_after_validator
            for stake_limit in (initial_stake_volume, math.inf):
                self.current_model_config['stake_limit'] = stake_limit
                for reward_type in ('constant', 'geometric'):
                    self.current_model_config['reward_type'] = reward_type
                    for total_reward in (1000, 10_000, 100_000):  # Varying load factor c in 0.1, 1, 10
                        self.current_model_config['total_rewards'] = total_reward
                        for gini_initial_distribution in (.0, .2, .4, .6, .8, .99999):
                            self.current_model_config['gini_initial_distribution'] = gini_initial_distribution
                            self.run_experiment()


def run_optimums():
    metrics = [Metrics.gini_stakes_diff,
               # Metrics.gini_rewards_diff,
               # Metrics.slope_gini_stakes,
               # Metrics.slope_gini_rewards
               ]
    optimums = Experiments(metrics).run()

    base_path = Path('results')
    if base_path.exists() and base_path.is_dir():
        shutil.rmtree(base_path)
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    data_visualization = DataVisualization()
    for metric in metrics:
        # TODO: Refactor duplicated code
        plots = data_visualization.run(history=optimums[metric]['min']['history'])
        if not os.path.exists(base_path / metric / 'min'):
            os.makedirs(base_path / metric / 'min')
        with open(base_path / metric / 'min' / "model_config.json", "w+") as fp:
            json.dump(optimums[metric]['min']['model_config'], fp, indent=4)
        for plot_name in plots.keys():
            plots[plot_name].savefig(base_path / metric / 'min' / plot_name)

        plots = data_visualization.run(history=optimums[metric]['max']['history'])
        if not os.path.exists(base_path / metric / 'max'):
            os.makedirs(base_path / metric / 'max')
        with open(base_path / metric / 'max' / "model_config.json", "w+") as fp:
            json.dump(optimums[metric]['max']['model_config'], fp, indent=4)
        for plot_name in plots.keys():
            plots[plot_name].savefig(base_path / metric / 'max' / plot_name)


def to_df(result_list):
    return pd.DataFrame.from_dict(
        [experiment['model_config'] | {'value': experiment['value']} for experiment in result_list])


def run_metrics_all_pos():
    metrics = [Metrics.gini_stakes_diff, Metrics.gini_rewards_diff]
    experiments = Experiments(metrics)
    experiments.run()
    return {metric: to_df(result_list) for metric, result_list in experiments.all_scores.items()}


def run_metrics_random_pos():
    metrics = [Metrics.gini_stakes_diff, Metrics.gini_rewards_diff]
    experiments = Experiments(metrics)
    experiments.run_random()
    return {metric: to_df(result_list) for metric, result_list in experiments.all_scores.items()}


def main():
    logging.basicConfig(level=logging.INFO)
    try:
        type = sys.argv[1]
    except:
        raise Exception("Insert a valid type in [coin_age, dynamic_weighted, random, weighted, all_pos]")
    logging.info("Executing " + type + " experiments")
    match type:
        case 'random':
            experiments_visualization = ExperimentsVisualization(run_metrics_random_pos())
            experiments_visualization.distance_from_optimum()
            experiments_visualization.summary()
            experiments_visualization.optimum()
            experiments_visualization.correlation()
            experiments_visualization.pairplot()


if __name__ == "__main__":
    main()
