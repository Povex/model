import os
import shutil
from pathlib import Path

from data_visualization import *
from metrics import Metrics
from experiments import Experiments
import json

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    logging.basicConfig(level=logging.INFO)
    metrics = [Metrics.gini_stakes_diff, Metrics.gini_rewards_diff, Metrics.slope_gini_stakes, Metrics.slope_gini_rewards]
    optimums = Experiments(metrics).run()

    base_path = Path('results')
    if base_path.exists() and base_path.is_dir():
        shutil.rmtree(base_path)
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    data_visualization = DataVisualization()
    for metric in metrics:
        # TODO: Refactor duplicated code
        plots = data_visualization.run(history=optimums[metric]['min']['history'])
        if not os.path.exists(base_path / metric / 'min'):
            os.makedirs(base_path / metric / 'min')
        with open(base_path / metric / 'min' / "model_config.json", "w+") as fp:
            json.dump(optimums[metric]['min']['model_config'], fp, indent=4)
        for plot_name in plots.keys():
            plots[plot_name].savefig(base_path / metric / 'min' / plot_name)

        plots = data_visualization.run(history=optimums[metric]['max']['history'])
        if not os.path.exists(base_path / metric / 'max'):
            os.makedirs(base_path / metric / 'max')
        with open(base_path / metric / 'max' / "model_config.json", "w+") as fp:
            json.dump(optimums[metric]['max']['model_config'], fp, indent=4)
        for plot_name in plots.keys():
            plots[plot_name].savefig(base_path / metric / 'max' / plot_name)


if __name__ == "__main__":
    main()
