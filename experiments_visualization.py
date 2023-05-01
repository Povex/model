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
            logging.info(f"Head for metric {metric}")
            df = df.apply(lambda x: np.abs(x) if x.name == 'value' else x) \
                .sort_values('value', ascending=True)
            logging.info(df)

    def mean_and_std_per_pos_type(self):
        for metric, df in self.metrics_df.items():
            logging.info(f"Mean and std for metric {metric}")
            logging.info(df.groupby(['pos_type']).agg({'value': ['mean', 'std']}).reset_index())

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
            df = df[['n_epochs', 'initial_stake_volume', 'total_rewards', 'reward_type', 'gini_initial_distribution',
                     'value']]
            plt.clf()
            sns.pairplot(df)
            plt.savefig(f'results/{metric}_pair_plot')

    def pairplot_reduction_factor(self):
        for metric, df in self.metrics_df.items():
            df = df[['n_epochs', 'initial_stake_volume', 'total_rewards', 'reward_type', 'coin_age_reduction_factor',
                     'gini_initial_distribution', 'value']]
            plt.clf()
            sns.pairplot(df)
            plt.savefig(f'results/{metric}_pair_plot')

    def pairplot_gini_threshold(self):
        for metric, df in self.metrics_df.items():
            df = df[['n_epochs', 'initial_stake_volume', 'total_rewards', 'reward_type', 'gini_threshold',
                     'gini_initial_distribution', 'value']]
            plt.clf()
            sns.pairplot(df)
            plt.savefig(f'results/{metric}_pair_plot')

    def correlation(self):
        for metric, df in self.metrics_df.items():
            logging.info(f"Correlation statistics for metric {metric}")
            df = df[['total_rewards', 'gini_initial_distribution', 'value']]
            plt.clf()
            corr_matrix = df.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
            plt.savefig(f'results/{metric}_correlation_matrix', bbox_inches='tight')

    def correlation_reduction_factor(self):
        for metric, df in self.metrics_df.items():
            logging.info(f"Correlation statistics for metric {metric}")
            df = df[['total_rewards', 'gini_initial_distribution', 'coin_age_reduction_factor', 'value']]
            plt.clf()
            corr_matrix = df.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
            plt.savefig(f'results/{metric}_correlation_matrix', bbox_inches='tight')

    def correlation_gini_threshold(self):
        for metric, df in self.metrics_df.items():
            logging.info(f"Correlation statistics for metric {metric}")
            df = df[['total_rewards', 'gini_initial_distribution', 'gini_threshold', 'value']]
            plt.clf()
            corr_matrix = df.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
            plt.savefig(f'results/{metric}_correlation_matrix', bbox_inches='tight')

    def geom_const_boxplot(self):
        for metric, df in self.metrics_df.items():
            plt.clf()
            plt.rcParams.update({'font.size': 22})
            logging.info(f"Constant and geometric boxplot for metric {metric}")
            dfs = [df.query("reward_type == 'constant'")['value'],
                   df.query("reward_type == 'geometric'")['value']]
            plt.boxplot(dfs, labels=['constant', 'geometric'])
            plt.savefig(f'results/{metric}_const_geom_boxplot')

    def metric_boxplot_grouped_by_pos_type(self):
        plt.clf()
        for metric, df in self.metrics_df.items():
            plt.clf()
            dfs = []
            x_values = []
            pos_types = df['pos_type'].unique()
            logging.info(f"Metric boxplot grouped by pos types {pos_types} for metric {metric}")
            for pos_type in pos_types:
                df_by_pos_type = df.query(f"pos_type == '{pos_type}'")
                summary = df_by_pos_type.describe()
                logging.info(f"{pos_type} summary", summary)
                dfs.append(df_by_pos_type['value'])
                x_values.append(pos_types)
            plt.boxplot(dfs, labels=x_values)
            plt.savefig(f'results/{metric}_by_pos_types')
