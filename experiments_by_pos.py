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


def get_default_model_config():
    return {
        "n_agents": 10,
        "n_epochs": 10000,
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


class ExperimentsByPos:
    """
    This class is utilized for analyzing individual PoS consensus mechanisms.
    Specifically, the PoS mechanism is fixed, while other parameters utilized in that PoS mechanism are varied.
    """

    def __init__(self, metrics: List[str], n_simulations=4, base_path=Path('random_pos_results')):
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

    def run_experiment(self):
        self.experiment_counter += 1
        logging.info(f"Current counter: {self.experiment_counter}")
        sim.ModelConfig().set_model_config(self.current_model_config)
        self.history = sim.ModelRunner.run(n_simulations=self.n_simulations)
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
        self.current_model_config['block_reward'] = 1
        for gini_initial_distribution in (.0, .2, .4, .6, .8, .999999999):
            self.current_model_config['gini_initial_distribution'] = gini_initial_distribution
            self.run_experiment()


def main():
    logging.basicConfig(level=logging.INFO)
    metrics = [Metrics.gini_stakes_diff, Metrics.gini_rewards_diff, Metrics.slope_gini_stakes,
               Metrics.slope_gini_rewards]
    ExperimentsByPos(metrics).run_random_pos_experiments()


if __name__ == "__main__":
    main()
