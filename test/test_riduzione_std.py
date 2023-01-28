import numpy as np
from simulator.model.statistics.statistics import *

data = np.array([100, 200, 600])
print("Dati prima della trasformazione:", data)
gini = gini_concentration_index(data)
print("Coefficiente di gini: ", gini)

print("Trafomazione")
data = data / data.sum()
print("Pesi:", data)


def linear_function(x1, y1, x2, y2):
    m = float(y2 - y1) / (x2 - x1)
    q = y1 - (m * x1)
    return m, q


def theta_function(x):
    m, q = linear_function(0.5, 0.005, 0.7, 0.1)
    return m * x + q


theta = 0
if gini >= 0.5:  # 0.5 è la threshold gini
    theta = theta_function(gini)


def reduce_gini_transform(data):
    m, q = linear_function(0.0, theta, 1, 1 - theta)
    return m * data + q


data = reduce_gini_transform(data)
data = data / data.sum() # La normalizzazione la fa già la libreria random
print("linear tranform", data)

print("Dati dopo la trasformazione:", data)
print("Coefficiente di gini: ", gini_concentration_index(data))
