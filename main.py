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

    experiments = Experiments().run()

    gini_stake_diff_min = math.inf
    gini_stake_diff_min_input = {'model_config': None, 'history': None}

    gini_stake_diff_max = -math.inf
    gini_stake_diff_max_input = {'model_config': None, 'history': None}


    for experiment in experiments:
        scores = Metrics(experiment['history']).scores(['gini_stakes_diff', 'gini_rewards_diff'])
        gini_stake_diff = scores['gini_stakes_diff']['mean']
        if gini_stake_diff < gini_stake_diff_min:
            gini_stake_diff_min = gini_stake_diff
            gini_stake_diff_min_input = experiment
        if gini_stake_diff > gini_stake_diff_max:
            gini_stake_diff_max = gini_stake_diff
            gini_stake_diff_max_input = experiment
    print(gini_stake_diff_min, gini_stake_diff_min_input)
    print(gini_stake_diff_max, gini_stake_diff_max_input)
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
