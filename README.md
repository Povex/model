# PoS simulator
This python project is an **agent-based simulator**
of variants of proof-of-stake (PoS) consensus algorithms that has been developed by exploiting
parallel computation techniques for better estimates and performance.
With this tool is possible to run **PoS stake model simulation** based on input parameters defined in the `model_config.json`.<br />

This project take part in my computer science thesis, where the goal is 
identify pos variations that are sustainable in the long term, ensuring a certain level of democracy in the system and a fair distribution of wealth, using metrics
based on the **Gini coefficient** and **Lorenz curves**.

## Table Of Content

- [Installation and setup](#setup)
- [Run a model](#run)
- [Graphs](#graphs)
- [License](#license)

## Installation and setup
The simulator is based on python3 (https://www.python.org/downloads/).
After the python installation, this project can be cloned with git:
- `git@github.com:Povex/model.git`

After the installation can be useful to setup a virtual environment:
- `python3 -m venv env`
And activate it:
- `source env/bin/activate`
The project dependencies can be installed with:
- `pip install -r requirements.txt`

The setup is done !

## Run a model
The parameters of a model can be setted modifying the `model_config.json` file, that can be found in the root of the project.
```jsonc
{
    "n_agents": 10, // Number of validators
    "n_epochs": 500000, // Number of epochs of each simulation
    /*pos_type can be "random", "weighted", "log_weighted", "dynamic_weighted", "inverse_weighted", "coin_age", "inverse_weighted_coin_age"*/
    "pos_type": "weighted",
    /*initial_distribution can be "constant", "linear", "polynomial", "custom", "gini"*/
    "initial_distribution": "gini",
    "gini_initial_distribution": 0.0,
    "log_weighted_base": 1.0, // If log_weighted pos_type is set then this paramenter should be setted 
    "log_shift_factor": 1.009, // If log_weighted pos_type is set then this paramenter should be setted 
    /*for a custom_distribution, the initial distribution should be setted to "custom" and the number of agents should be the same of this custum_distributions elemnts number*/
    "custom_distribution": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "initial_stake_volume": 1000.0, // Volume of the initial state of the simulation (the first epoch)
    "reward_type": "constant", // Can be "constant" or "geometric"
    "total_rewards": 1000.0, // Total reward dispensed through the entire simulation
    "gini_threshold": 0.7, // Parameter used in "dynamic_weighted" pos_type
    "dynamic_weighted_theta": 0.4, // Parameter used in "dynamic_weighted" pos_type
    "malicious_node_probability": 0.0, // Set the probability of a node to be malicious 
    "percent_stake_penalty": 0.2, // Set the percentage of stake loss if a node is malicious and is choosen as validator
    "stop_epoch_after_malicious": 5, // After a node is found to be malicious, it cannot be choosen as validator for this amount of epochs
    "stop_epoch_after_validator": 0, // After a node is choosen as validator, it cannot be choosen as again validator for this amount of epochs
    "coin_age_reduction_factor": 0.5, // If a node is choosen as validator and the pos_type is "coin_age" then the coin_age is reduced by this percentage
    "stake_limit": 999999999999999.0, // If this limit is reached for an agent a_i, then nothing can be added to the stake of the agent a_i
    "min_coin_age": 0,  // The minumum coin age that is needed to take part of the validators
    "max_coin_age": 999999999999999, // The maximum coin age that can be reached
    "early_withdrawing_penalty": 0.2, // TODO
    "weighted_reward": 1.0 // TODO
}
```
A model configuration can also be setted at run time, using the singleton object of type `ModelConfig`:
- `custom_config = {n_agents": 10, "n_epochs": 1000,"initial_stake_volume": 1000.0,.....}`
- `sim.ModelConfig(custom_config)` 

After the needed paramenters are choosen, the model can be runned with the following instructions:
- From the project root directory: `python3` to get the python3 REPL:
- Import the simulator library with `import simulator as sim`
- Set the number of parallel simulations: `n_parallel_simulations = 2`
- Run the model and obtain an history `history = sim.ModelRunner.run(n_parallel_simulations)`

The `history` object obtained is a pandas dataframe that contains all the epochs runned for all the parallel simulation setted.
The `history` object can be then analyzed, for example, can be obtained the Gini concentration index of an epoch executing the `gini_concentration_index\1` function inside the `statistics` package:
- `from simulator.model.statistics.statistics import *`
- `epoch = 42`
- `gini_index = history.query(f'epoch == {epoch}') \
            .drop(columns=['id', 'epoch']) \
            .groupby(['simulation'])["stake"] \
            .apply(gini_concentration_index) \
            .reset_index()`

More example of statistic functions can be found in the Metric class in `metrics.py`.
## Graphs
Graphs can be generated using matplotlib. <br> 
A set of predefined graphs are present inside `data_visualization.py` in DataVisualization class.
This class take in input a history dataframe to show Some useful plots, for example:
- stakes_gini_index
- stakes_gini_ratio
- rewards_gini_index
- rewards_gini_ratio
- lorenz_curves
- first_last_stakes_hist
- stakes_for_agents
- mean_and_std_stakes

## License
