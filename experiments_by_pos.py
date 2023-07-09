import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

import simulator as sim
from data_visualization import DataVisualization
from matplotlib import pyplot as plt

from simulator.model.statistics.statistics import gini_concentration_index


def get_default_model_config():
    return {
        "n_agents": 10,
        "n_epochs": 1000,
        "initial_stake_volume": 1000.0,
        "total_rewards": 1000,
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


class ExperimentsByPoS:
    """
    This class is utilized for analyzing individual PoS consensus mechanisms.
    Specifically, the PoS mechanism is fixed, while other parameters utilized in that PoS mechanism are varied.
    """

    def __init__(self, n_simulations=4, base_path=Path('results/pos_results')):
        super().__init__()
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
        return ginis['gini']['mean'], ginis['gini']['std']

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
        return ginis['gini']['mean'], ginis['gini']['std']

    def plot_ginis_by_initial_distribution(self):
        cmap = plt.get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, len(self.ginis_by_initial_distributions)))
        fig = plt.figure()
        x = self.epochs
        for i, y in enumerate(self.ginis_by_initial_distributions):
            color = colors[i]
            y_mean = y[1][0]
            y_std_upper = y_mean + y[1][1]
            y_std_lower = y_mean - y[1][1]
            plt.plot(x, y_mean, label=f"Initial Gini: {y[0]}", linestyle="-", color=color)
            plt.plot(x, y_std_upper, linestyle="-.", linewidth=0.5, alpha=0.1)
            plt.plot(x, y_std_lower, linestyle="-.", linewidth=0.5, alpha=0.1)
            plt.fill_between(x=x.values, y1=y_std_upper.values, y2=y_std_lower.values, alpha=.1, color=color)
        plt.grid()
        plt.legend(loc='upper right', framealpha=0.2)
        plt.title("Stake distributions", fontsize=17)
        plt.ylabel("Gini concentration index", fontsize=14)
        plt.xlabel("Time [epochs]", fontsize=14)
        fig.savefig(self.base_path / 'ginis_by_initial_distributions')

    def plot_ginis_rewards_by_initial_distribution(self):
        cmap = plt.get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, len(self.ginis_rewards_by_initial_distributions)))
        fig = plt.figure()
        x = self.epochs
        for i, y in enumerate(self.ginis_rewards_by_initial_distributions):
            color = colors[i]
            y_mean = y[1][0]
            y_std_upper = y_mean + y[1][1]
            y_std_lower = y_mean - y[1][1]
            plt.plot(x, y_mean, label=f"Initial Gini: {y[0]}", linestyle="-", color=color)
            plt.plot(x, y_std_upper, linestyle="-.", linewidth=0.5, alpha=0.1)
            plt.plot(x, y_std_lower, linestyle="-.", linewidth=0.5, alpha=0.1)
            plt.fill_between(x=x.values, y1=y_std_upper.values, y2=y_std_lower.values, alpha=.1, color=color)
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

    def run_random_pos_experiments(self):
        self.current_model_config['pos_type'] = 'random'
        self.current_model_config['initial_distribution'] = 'gini'
        for reward_function in ('constant',):  # 'geometric'):
            self.current_model_config['reward_type'] = reward_function
            for gini_initial_distribution in (.0, .2, .4, .6, .8, .999999999):
                self.current_model_config['gini_initial_distribution'] = gini_initial_distribution
                self.run_experiment()

    def run_coin_age_experiments(self):
        self.current_model_config['pos_type'] = 'coin_age'
        self.current_model_config['initial_distribution'] = 'gini'
        for reward_function in ('constant',):  # 'geometric'):
            self.current_model_config['reward_type'] = reward_function
            for gini_initial_distribution in (.0, .2, .4, .6, .8, .999999999):
                self.current_model_config['gini_initial_distribution'] = gini_initial_distribution
                self.run_experiment()

    def run_dynamic_weighted_experiments(self):
        self.current_model_config['pos_type'] = 'dynamic_weighted'
        self.current_model_config['initial_distribution'] = 'gini'
        self.current_model_config['gini_threshold'] = 0.4
        for reward_function in ('constant',):  # 'geometric'):
            self.current_model_config['reward_type'] = reward_function
            for gini_initial_distribution in (.0, .2, .4, .6, .8, .999999999):
                self.current_model_config['gini_initial_distribution'] = gini_initial_distribution
                self.run_experiment()

    def run_weighted_pos_experiments(self):
        self.current_model_config['pos_type'] = 'weighted'
        self.current_model_config['initial_distribution'] = 'gini'
        for reward_function in ('geometric',): #'geometric'):
            self.current_model_config['reward_type'] = reward_function
            for gini_initial_distribution in (.0, .2, .4, .6, .8, .999999999):
                self.current_model_config['gini_initial_distribution'] = gini_initial_distribution
                self.run_experiment()


def main():
    logging.basicConfig(level=logging.INFO)
    try:
        pos_type = sys.argv[1]
    except:
        raise Exception("Insert a pos type in [coin_age, dynamic_weighted, random, weighted]")

    logging.info("Executing" + pos_type + "PoS experiments")
    match pos_type:
        case "coin_age":
            pos_experiments = ExperimentsByPoS(base_path=Path('results/coin_age_results'))
            pos_experiments.run_coin_age_experiments()
        case "dynamic_weighted":
            pos_experiments = ExperimentsByPoS(base_path=Path('results/dynamic_weighted_results'))
            pos_experiments.run_dynamic_weighted_experiments()
        case "random":
            pos_experiments = ExperimentsByPoS(base_path=Path('results/random_results'))
            pos_experiments.run_random_pos_experiments()
        case "weighted":
            pos_experiments = ExperimentsByPoS(base_path=Path('results/weighted_results'))
            pos_experiments.run_weighted_pos_experiments()
        case _:
            raise Exception("Insert a pos type in [coin_age, dynamic_weighted, random, weighted]")
    pos_experiments.plot_ginis_by_initial_distribution()
    pos_experiments.plot_ginis_rewards_by_initial_distribution()


if __name__ == "__main__":
    main()
