import logging
import math
import warnings
import simulator as sim
from data_visualization import *
from metrics import Metrics
from experiments import Experiments

warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    logging.basicConfig(level=logging.INFO)
    metrics = [Metrics.gini_stakes_diff, Metrics.gini_rewards_diff]
    optimums = Experiments(metrics).run()
    print(optimums)
    # Calcolare gli ottimi dei vari scores e interpretarli visualmente con il data visualization


    # score = Metrics(history).scores(['gini_stakes_diff'])

    # score(history, metric='gini_diff')
    # position_indexes(history)
    # dispersions_indexes(history)
    # visualization_gini_rcg(history)
    # gini_on_stake_rewards(history)
    # stake_histogram(history)
    # prova_histogram_3d(history)
    # lorenz_curves_3d(history)
    # stake_histogram_normalized(history)
    # stake_histogram_evolution(history)
    # time_series_histogram(history)

if __name__ == "__main__":
    main()
