import random

import numpy as np

from simulator.model.proof_of_stake.proof_of_stake import PoS
from simulator.model.statistics.statistics import gini_concentration_index
from scipy.optimize import minimize


class DynamicWeighted(PoS):
    """
    Il sistema dinamico permette di modificare il modello al variare di certe condizioni, per esempio al raggiungimento
    di una soglia nell'indice di Gini.
     """

    def __init__(self):
        super().__init__()
        self.create_agents()

    def linear_function(self, x1, y1, x2, y2):
        m = float(y2 - y1) / (x2 - x1)
        q = y1 - (m * x1)
        return m, q

    def theta_function(self, x):
        m, q = self.linear_function(0.5, 0.005, 0.7, 0.1)  # reduction gini to 0.5
        return m * x + q

    def reduce_gini_transform(self, data, theta):
        m, q = self.linear_function(0.0, theta, 1, 1 - theta)
        return m * data + q

    def weights_transformation(self, weights, original_gini):
        theta = 0
        if original_gini >= self.conf.gini_threshold:
            theta = self.theta_function(original_gini)
        transformed_data = self.reduce_gini_transform(weights, theta)
        # data = data / data.sum()  # data already normalized by the random library
        return transformed_data

    def dynamic_gini_weighted(self):
        agents = list(self.agents.difference(self.agents_stop_epochs))
        data = np.array([a.stake for a in self.agents])
        weights = data / data.sum()
        gini = gini_concentration_index(weights)
        transformed_data = self.weights_transformation(weights, gini)
        return random.choices(agents, weights=transformed_data)[0]

    def select_validator(self):
        return self.dynamic_gini_weighted()
