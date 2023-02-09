import json
import logging
import math
import os
import shutil
from pathlib import Path
from typing import List

import simulator as sim
from data_visualization import DataVisualization
from metrics import Metrics
from matplotlib import pyplot as plt

from simulator.model.statistics.statistics import gini_concentration_index


def get_default_model_config():
    return {
        "n_agents": 10,
        "n_epochs": 50000,
        "initial_stake_volume": 1000.0,
        "total_rewards": 50000.0,
        'reward_type': 'constant',

        "stop_epoch_after_validator": 0,
        "stake_limit": 999999999999999.0,

        "malicious_node_probability": 0.0,
        "stop_epoch_after_malicious": 5,
        "percent_stake_penalty": 0.2,

        "pos_type": "weighted",
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


class ExperimentsByWeightedPos:
    """
    This class is utilized for analyzing individual PoS consensus mechanisms.
    Specifically, the PoS mechanism is fixed, while other parameters utilized in that PoS mechanism are varied.
    """

    def __init__(self, metrics: List[str], n_simulations=4, base_path=Path('results/weighted_pos_results')):
        super().__init__()
        self.metrics = metrics
        self.current_model_config = get_default_model_config()
        sim.ModelConfig(self.current_model_config)
        self.scores = None
        self.history = None
        self.n_simulations = n_simulations
        self.base_path = base_path
        self.experiment_counter: int = 0
        self.data_visualization = DataVisualization()
        self.ginis_by_initial_distributions = []
        self.ginis_rewards_by_initial_distributions = []
        self.epochs = None

    def calculate_ginis(self):
        ginis = self.history.drop(columns=['id']) \
            .groupby(['epoch', 'simulation'])['stake'] \
            .apply(gini_concentration_index) \
            .reset_index(name='gini') \
            .groupby('epoch') \
            .agg({'gini': ['mean', 'std']}) \
            .reset_index()
        self.epochs = ginis['epoch']
        return ginis['gini']['mean']

    def calculate_ginis_rewards(self):
        initial_stakes = self.history.query("epoch == 0").query("simulation == 0")
        history = self.history.merge(initial_stakes, on='id', suffixes=('', '_')) \
            .eval("stake = stake - stake_") \
            .drop(columns=['stake_'])
        ginis = history.drop(columns=['id']) \
            .groupby(['epoch', 'simulation'])['stake'] \
            .apply(gini_concentration_index) \
            .reset_index(name='gini') \
            .groupby('epoch') \
            .agg({'gini': ['mean', 'std']}) \
            .reset_index()
        return ginis['gini']['mean']

    def plot_ginis_by_initial_distribution(self):
        fig = plt.figure()
        x = self.epochs
        for y in self.ginis_by_initial_distributions:
            plt.plot(x, y[1], label=f"Initial Gini: {y[0]}", linestyle="-")
        plt.grid()
        plt.legend(loc='upper right')
        plt.title("Stake distributions", fontsize=17)
        plt.ylabel("Gini concentration index", fontsize=14)
        plt.xlabel("Time [epochs]", fontsize=14)
        fig.savefig(self.base_path / 'ginis_by_initial_distributions')

    def plot_ginis_rewards_by_initial_distribution(self):
        fig = plt.figure()
        x = self.epochs
        for y in self.ginis_rewards_by_initial_distributions:
            plt.plot(x, y[1], label=f"Initial Gini: {y[0]}", linestyle="-")
        plt.grid()
        plt.legend(loc='upper right')
        plt.title("Rewards distributions", fontsize=17)
        plt.ylabel("Gini concentration index", fontsize=14)
        plt.xlabel("Time [epochs]", fontsize=14)
        fig.savefig(self.base_path / 'ginis_rewards_by_initial_distributions')

    def run_experiment(self):
        self.experiment_counter += 1
        logging.info(f"Current counter: {self.experiment_counter}")
        sim.ModelConfig().set_model_config(self.current_model_config)
        self.history = sim.ModelRunner.run(n_simulations=self.n_simulations)
        self.ginis_by_initial_distributions.append(
            (round(sim.ModelConfig().gini_initial_distribution, 1), self.calculate_ginis()))
        self.ginis_rewards_by_initial_distributions.append(
            (round(sim.ModelConfig().gini_initial_distribution, 1), self.calculate_ginis_rewards()))
        # self.scores = Metrics(self.history).scores(self.metrics)
        if not os.path.exists(self.base_path / str(self.experiment_counter)):
            os.makedirs(self.base_path / str(self.experiment_counter))
        with open(self.base_path / str(self.experiment_counter) / "model_config.json", "w+") as fp:
            json.dump(self.current_model_config, fp, indent=4)
        plots = self.data_visualization.run(history=self.history)
        for plot_name in plots.keys():
            plots[plot_name].savefig(self.base_path / str(self.experiment_counter) / plot_name)

    def run_weighted_pos_experiments(self):
        self.current_model_config['pos_type'] = 'weighted'
        self.current_model_config['initial_distribution'] = 'gini'
        for reward_function in ('constant',): #'geometric'):
            self.current_model_config['reward_type'] = reward_function
            for gini_initial_distribution in (.0, .2, .4, .6, .8, .999999999):
                self.current_model_config['gini_initial_distribution'] = gini_initial_distribution
                self.run_experiment()


def main():
    logging.basicConfig(level=logging.INFO)
    metrics = [Metrics.gini_stakes_diff, Metrics.gini_rewards_diff, Metrics.slope_gini_stakes,
               Metrics.slope_gini_rewards]
    random_pos_experiments = ExperimentsByWeightedPos(metrics)
    random_pos_experiments.run_weighted_pos_experiments()
    random_pos_experiments.plot_ginis_by_initial_distribution()
    random_pos_experiments.plot_ginis_rewards_by_initial_distribution()


if __name__ == "__main__":
    main()
