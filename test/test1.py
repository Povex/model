
import simulator as sim

model_config = {'n_agents': 10, 'n_epochs': 100, 'initial_stake_volume': 1000.0, 'block_reward': 1.0, 'stop_epoch_after_validator': 5, 'stake_limit': 500, 'malicious_node_probability': 0.0, 'stop_epoch_after_malicious': 5, 'percent_stake_penalty': 0.2, 'pos_type': 'inverse_weighted', 'initial_distribution': 'gini', 'gini_initial_distribution': 0.8, 'custom_distribution': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'gini_threshold': 0.8, 'log_weighted_base': 2, 'log_shift_factor': 1, 'min_coin_age': 0, 'max_coin_age': 999999999999999, 'early_withdrawing_penalty': 0.2, 'weighted_reward': 1.0}

sim.ModelConfig(model_config)
history = sim.ModelRunner.run(4)
print(history.query("epoch == 0"))
print(history.query("epoch == 100"))

# TODO: Dato che lo stake limit è impostato a 500, quello che succede è che la distribuzione iniziale non ha davvero un gini 0.8, ma di meno.. infatti il più ricco a 820 viene limitato a 500, come lo gestisco ?
# TODO: Quando si raggiunge il limite massimo di stake, dove vanno a finire i reward ? Vanno persi, è corretto ? Si perchè nella realtà i reward vengono messi sul conto ma non "investiti" nello stake.
