import warnings
import simulator as sim
from simulation_visualization import visualization_gini

warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    model_config = {
        "n_agents": 10,
        "n_epochs": 100_000,
        "pos_type": "random",
        "initial_distribution": "constant",
        "gini_initial_distribution": 0.9,
        "custom_distribution": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "initial_stake_volume": 1000.0,
        "block_reward": 1.0,
        "gini_threshold": 0.8,
        "log_weighted_base": 2,
        "log_shift_factor": 1,
        "malicious_node_probability": 0.0,
        "percent_stake_penalty": 0.2,
        "stop_epoch_after_malicious": 5,
        "stop_epoch_after_validator": 0,
        "stake_limit": 999999999999999.0,
        "min_coin_age": 0,
        "max_coin_age": 999999999999999,
        "early_withdrawing_penalty": 0.2,
        "weighted_reward": 1.3
    }
    sim.ModelConfig(model_config)
    n_simulations = 4
    history = sim.ModelRunner.run(n_simulations=n_simulations)
    visualization_gini(history)


if __name__ == "__main__":
    main()
