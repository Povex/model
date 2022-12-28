import simulator as sim

model_config = {
    "n_agents": 10,
    "n_epochs": 1000,
    "pos_type": "random",
    "initial_distribution": "gini",
    "gini": 1.0,
    "custom_distribution": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "initial_stake_volume": 1000.0,
    "block_reward": 1.0,
    "malicious_node_probability": 0.0,
    "percent_stake_penalty": 0.2,
    "stop_epoch_after_malicious": 5,
    "stop_epoch_after_validator": 5,
    "stake_limit": 999999999999999.0,
    "min_coin_age": 0,
    "max_coin_age": 999999999999999,
    "early_withdrawing_penalty": 0.2,
    "weighted_reward": 1.3
}

sim.ModelConfig(model_config)
history = sim.ModelRunner.run(1)
print(history[0][0])
