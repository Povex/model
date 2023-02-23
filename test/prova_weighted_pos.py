import random

from simulator.model.statistics.statistics import *

agents = [
    {'id': 0, 'stake': 1},
    {'id': 1, 'stake': 1},
    {'id': 2, 'stake': 1},
    {'id': 3, 'stake': 1},
    {'id': 4, 'stake': 1},
    {'id': 5, 'stake': 1},
    {'id': 6, 'stake': 1},
    {'id': 7, 'stake': 1},
    {'id': 8, 'stake': 1},
    {'id': 9, 'stake': 100},
]

print(gini_concentration_index([a['stake'] for a in agents]))

for i in range(1000):
    validator = random.choices(agents, weights=[a['stake'] for a in agents])[0]
    validator['stake'] += 5
    print(f"{i}:", gini_concentration_index([a['stake'] for a in agents]))

