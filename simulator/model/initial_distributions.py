import math

import numpy as np

from simulator.model.exceptions.exception_model_config import ModelConfigException
from simulator.model.model_config import ModelConfig


class Distributions:

    @staticmethod
    def generate_distribution(dist_type: str, n: int, volume: float):
        match dist_type:
            case "constant":
                return Distributions.constant(n, volume)
            case "linear":
                return Distributions.linear(n, volume)
            case "polynomial":
                return Distributions.polynomial(n, volume, 100)
            case "custom":
                return ModelConfig().custom_distribution
            case "gini":
                return Distributions.gini(n, volume, ModelConfig().gini_initial_distribution)
            case _:
                raise ModelConfigException(f"No initial distribution found for {dist_type}")

    @staticmethod
    def constant(n_agents, stake_volume):
        stake = stake_volume / n_agents
        return [stake for _ in range(n_agents)]

    @staticmethod
    def gini(n: int, volume: float, gini: float):
        def lorenz_curve(x1, y1, x2, y2):
            m = float(y2 - y1) / (x2 - x1)
            return lambda x: m * x

        max_r = (n - 1) / 2
        r = gini * max_r # Mantengo R
        prop = ((n - 1) / n) * ((max_r - r) / max_r)
        lc = lorenz_curve(0, 0, (n - 1) / n, prop)
        q = [lc(i / n) for i in range(1, n)] + [1]
        cumulate_sum = [i * volume for i in q]
        stakes = [cumulate_sum[0]] + [cumulate_sum[i] - cumulate_sum[i - 1] for i in range(1, n)]
        return stakes

    @staticmethod
    def constant_np(n_agents, stake_volume):
        stake = stake_volume / n_agents
        return np.full(n_agents, stake)

    @staticmethod
    def linear(n_agents, stake_volume):
        m = stake_volume / (.5 * (math.pow(n_agents, 2) + n_agents))
        stakes = []
        for i in range(1, n_agents + 1):
            stakes.append(m * i)
        return stakes

    @staticmethod
    def polynomial(n_agents, stake_volume, degree=2):
        sum = 0
        for i in range(1, n_agents + 1):
            sum += math.pow(i, degree)
        m = stake_volume / sum
        stakes = []
        for i in range(1, n_agents + 1):
            stakes.append(m * math.pow(i, degree))
        return stakes
