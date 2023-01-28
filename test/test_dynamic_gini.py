# demonstration of the power transform on data with a skew
import numpy as np
from numpy import array
from simulator.model.statistics.statistics import gini_concentration_index
from scipy.optimize import minimize

data = np.array([3, 6, 30, 1, 1000])
weights = data/sum(data)
print("Data:", data)
print("Gini data:", gini_concentration_index(data))
print("Weights:", weights)
print("Gini data:", gini_concentration_index(weights))


# calcolare il coefficiente di Gini originale
original_gini = gini_concentration_index(weights)

# definire la funzione obiettivo come la differenza tra il coefficiente di Gini originale e la metà del coefficiente di Gini originale
def objective_func(x):
    return abs(gini_concentration_index(x) - original_gini / 2)

# definire le restrizioni, ad esempio la somma delle probabilità deve essere uguale a 1 e l'ordinamento deve essere mantenuto
def constraint_func_sum(x):
    return 1 - x.sum()

def constraint_func_order(x):
    return (x - weights).max()

# definire i vincoli di non negatività per ogni elemento della distribuzione di probabilità
bnds = [(0, None) for _ in range(len(weights))]

# eseguire l'ottimizzazione
result = minimize(objective_func, weights, bounds=bnds, constraints=[{"fun": constraint_func_sum, "type": "eq"}, {"fun":constraint_func_order, "type":"ineq"}])

# stampa la distribuzione ottimizzata
print(result.x)
print(gini_concentration_index(result.x))



# def linear_function(x1, y1, x2, y2):
#     m = float(y2 - y1) / (x2 - x1)
#     q = y1 - (m * x1)
#     return m, q
#
# def coeff(gini):
#     m, q = linear_function(0.1, 1, 1, 5)
#     return m * gini + q
#
# gini = gini_concentration_index(data)
# if gini >= 0.3:
#     rad_index = coeff(gini_concentration_index(weights))
#     data = data ** (1 / rad_index)
# print(data)
# print("Gini:", gini_concentration_index(data))
