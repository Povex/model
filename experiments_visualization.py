import logging
from typing import Dict
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class ExperimentsVisualization:

    def __init__(self, metrics_df):
        self.metrics_df: Dict[str, pd.DataFrame] = metrics_df

    def distance_from_optimum(self):
        for metric, df in self.metrics_df.items():
            print(f"Head for metric {metric}")
            df = df.apply(lambda x: np.abs(x) if x.name == 'value' else x) \
                .sort_values('value', ascending=True)
            print(df)

    def mean_and_std_per_pos_type(self):
        for metric, df in self.metrics_df.items():
            print(f"Mean and std for metric {metric}")
            print(df.groupby(['pos_type']).agg({'value': ['mean', 'std']}).reset_index())

    def summary(self):
        plt.clf()
        dfs = []
        metrics = []
        for metric, df in self.metrics_df.items():
            logging.info(f"Summary statistics for metric {metric} distance")
            summary = df.describe()
            logging.info(summary)
            dfs.append(df['value'])
            metrics.append(metric)
        plt.boxplot(dfs, labels=metrics)
        plt.savefig(f'results/boxplot_summary')

    def optimum(self):
        """
        :return: The model config that have the abs(value) of the metric closer to 0.
        """
        for metric, df in self.metrics_df.items():
            logging.info(f"Optimum statistics for metric {metric} distance")
            distance_df = df.apply(lambda x: np.abs(x) if x.name == 'value' else x).sort_values('value', ascending=True)
            logging.info(distance_df)
            optimum = distance_df.iloc[0]
            logging.info(optimum)

    def pairplot(self):
        for metric, df in self.metrics_df.items():
            df = df[['n_epochs', 'initial_stake_volume', 'total_rewards', 'reward_type', 'gini_initial_distribution', 'stop_epoch_after_validator', 'value']]
            plt.clf()
            sns.pairplot(df)
            plt.savefig(f'results/{metric}_pair_plot')

    def correlation(self):
        for metric, df in self.metrics_df.items():
            logging.info(f"Correlation statistics for metric {metric}")
            df = df[['total_rewards', 'gini_initial_distribution', 'stop_epoch_after_validator', 'value']]
            plt.clf()
            corr_matrix = df.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
            plt.savefig(f'results/{metric}_correlation_matrix', bbox_inches='tight')

