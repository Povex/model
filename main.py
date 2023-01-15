import logging
import math
import warnings
import simulator as sim
from data_visualization import *
from metrics import Metrics
from experiments import Experiments

warnings.simplefilter(action='ignore', category=FutureWarning)


#
def main():
    logging.basicConfig(level=logging.INFO)
    metrics = [Metrics.gini_stakes_diff, Metrics.gini_rewards_diff]
    optimums = Experiments(metrics).run()
    print(optimums)
    # Calcolare gli ottimi dei vari scores e interpretarli visualmente con il data visualization
    # Per ogni punto di ottimo esegui le data visualization e salva i risultati

    base_path = Path('results')
    if base_path.exists() and base_path.is_dir():
        shutil.rmtree(base_path)
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    plots = DataVisualization(history=optimums[Metrics.gini_stakes_diff]['min']['history']).run()
    for plot_name in plots.keys():
        if not os.path.exists(base_path / Metrics.gini_stakes_diff):
            os.makedirs(base_path / Metrics.gini_stakes_diff)
        plots[plot_name].savefig(base_path / Metrics.gini_stakes_diff / plot_name)

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
