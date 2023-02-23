from simulator.model.statistics.statistics import gini_concentration_ratio, gini_concentration_index


x = [100, 100, 100, 100, 100, 500]
print("Index: ", gini_concentration_index(x))
print("Ratio: ", gini_concentration_ratio(x))