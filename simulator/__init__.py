import pandas as pd

from simulator.model.proof_of_stake.coin_age_proof_of_stake import CoinAgePoS
from simulator.model.proof_of_stake.dynamic_weighted_coin_age_proof_of_stake import DynamicWeightedCoinAge
from simulator.model.proof_of_stake.dynamic_weighted_proof_of_stake import DynamicWeighted
from simulator.model.proof_of_stake.inverse_weighted import InverseWeightedPoS
from simulator.model.proof_of_stake.log_weighted_proof_of_stake import LogWeightedPoS
from simulator.model.proof_of_stake.random_proof_of_stake import RandomPoS
from simulator.model.proof_of_stake.weighted_proof_of_stake import WeightedPoS

import multiprocessing as mp

from simulator.model.model_config import ModelConfig


class ModelRunner:

    @staticmethod
    def model_selector():
        pos_type = ModelConfig().pos_type
        match pos_type:
            case "random":
                return RandomPoS()
            case "weighted":
                return WeightedPoS()
            case "log_weighted":
                return LogWeightedPoS()
            case "dynamic_weighted":
                return DynamicWeighted()
            case "inverse_weighted":
                return InverseWeightedPoS()
            case "coin_age":
                return CoinAgePoS()
            case "inverse_weighted_coin_age":
                return DynamicWeightedCoinAge()
            case _:
                raise Exception(f"No model matches the type {pos_type}")

    @staticmethod
    def run_model(return_list):
        model = ModelRunner.model_selector()
        model.run()
        return_list.append(model.history)

        return return_list

    @staticmethod
    def run(n_simulations=1, n_processors=mp.cpu_count()):
        pool = mp.Pool(n_processors)
        manager = mp.Manager()
        return_list = manager.list()

        for _ in range(n_simulations):
            pool.apply_async(ModelRunner.run_model, args=(return_list,))
        pool.close()
        pool.join()

        # Concatenation of the simulations in one dataset
        compact_history = [return_list[i].assign(simulation=i) for i in range(len(return_list))]
        compact_history = pd.concat(compact_history, ignore_index=True)

        return compact_history
