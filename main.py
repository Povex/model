import os
import shutil
from pathlib import Path

from data_visualization import *
from metrics import Metrics
from experiments import Experiments

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    logging.basicConfig(level=logging.INFO)
    metrics = [Metrics.gini_stakes_diff, Metrics.gini_rewards_diff]
    optimums = Experiments(metrics).run()

    base_path = Path('results')
    if base_path.exists() and base_path.is_dir():
        shutil.rmtree(base_path)
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    data_visualization = DataVisualization()
    for metric in metrics:
        plots = data_visualization.run(history=optimums[metric]['min']['history'])
        for plot_name in plots.keys():
            if not os.path.exists(base_path / metric):
                os.makedirs(base_path / metric)
            plots[plot_name].savefig(base_path / metric / plot_name)


if __name__ == "__main__":
    main()
