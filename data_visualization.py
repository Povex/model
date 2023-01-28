import logging

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter

from simulator.model.statistics.statistics import *


class DataVisualization:

    def __init__(self):
        self.history: pd.DataFrame = pd.DataFrame()
        self.plots = {}

    def run(self, history: pd.DataFrame):
        logging.info("Started run data visualization")
        self.history = history
        self.plots['stakes_gini_index'] = self.__stakes_gini_index()
        #self.plots['stakes_gini_ratio'] = self.__stakes_gini_ratio()
        self.plots['rewards_gini_index'] = self.__rewards_gini_index()
        #self.plots['rewards_gini_ratio'] = self.__rewards_gini_ratio()
        self.plots['lorenz_curves'] = self.__lorenz_curves_3d()
        self.plots['first_last_stakes_hist'] = self.__stake_histogram()
        self.plots['stakes_for_agents'] = self.__stakes_for_agents()
        self.plots['mean_and_std_stakes'] = self.__mean_and_std_stakes()
        return self.plots

    def __stakes_gini_index(self):
        """
        Data visualization of Gini concentration index in function of the epochs
        """
        history = self.history.drop(columns=['id']) \
            .groupby(['epoch', 'simulation'])['stake'] \
            .apply(gini_concentration_index) \
            .reset_index(name='gini') \
            .groupby('epoch') \
            .agg({'gini': ['mean', 'std']}) \
            .reset_index()

        fig = plt.figure()
        x = history['epoch']
        y_mean = history['gini']['mean']
        y_std_upper = y_mean + history['gini']['std']
        y_std_lower = y_mean - history['gini']['std']
        plt.plot(x, y_mean, label="mean ± std.dev", color='blue', linestyle="-")
        plt.plot(x, y_std_upper, color='lightsteelblue', linestyle="-.", linewidth=0.5, alpha=0.1)
        plt.plot(x, y_std_lower, color='lightsteelblue', linestyle="-.", linewidth=0.5, alpha=0.1)
        plt.fill_between(x=x.values, y1=y_std_upper.values, y2=y_std_lower.values, alpha=.1, color="blue")
        plt.grid()
        plt.legend()
        plt.title("Stake distributions", fontsize=17)
        plt.ylabel("Gini concentration index", fontsize=14)
        plt.xlabel("Time [epochs]", fontsize=14)
        # plt.savefig(self.base_path / 'gini.png')
        return fig

    def __stakes_gini_ratio(self):
        """
        Data visualization of Gini concentration ratio in function of the epochs
        """
        history = self.history.drop(columns=['id']) \
            .groupby(['epoch', 'simulation'])['stake'] \
            .apply(gini_concentration_ratio) \
            .reset_index(name='gini') \
            .groupby('epoch') \
            .agg({'gini': ['mean', 'std']}) \
            .reset_index()

        fig = plt.figure()
        x = history['epoch']
        y_mean = history['gini']['mean']
        y_std_upper = y_mean + history['gini']['std']
        y_std_lower = y_mean - history['gini']['std']
        plt.plot(x, y_mean, label="mean ± std.dev", color='blue', linestyle="-")
        plt.plot(x, y_std_upper, color='lightsteelblue', linestyle="-.", linewidth=0.5, alpha=0.1)
        plt.plot(x, y_std_lower, color='lightsteelblue', linestyle="-.", linewidth=0.5, alpha=0.1)
        plt.fill_between(x=x.values, y1=y_std_upper.values, y2=y_std_lower.values, alpha=.1, color="blue")
        plt.grid()
        plt.legend()
        plt.title("Stake distributions", fontsize=17)
        plt.ylabel("Gini concentration ratio", fontsize=14)
        plt.xlabel("Time [epochs]", fontsize=14)
        # plt.savefig(self.base_path / 'gini.png')
        return fig

    def __lorenz_curves_3d(self):
        # TODO: aggregate simulation stakes for lorenz curves
        # TODO: how i can show the std ?
        history = self.history.drop(columns=['id'])

        last_epoch = history["epoch"].max()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        def plot2din3d(x, y, z):
            ax.plot(x, y, zs=z, color='royalblue')
            ax.plot(x, x, z, color='black', alpha=.1)

        n_planes = 10
        steps = int((last_epoch + 1) / n_planes)
        for z in range(0, last_epoch + 1, steps):
            f, q = lorenz_curve(history.query(f"epoch == {z}").query("simulation == 0")["stake"])
            plot2din3d(f, q, z)

        # ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(last_epoch, 0)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Epochs')

        ax.view_init(elev=110., azim=-72.0)

        return fig

    def __stake_histogram(self):
        # TODO: std deviation of this ?
        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.grid(axis='y')
        plt.legend()
        plt.title("First epoch", fontsize=14)
        plt.ylabel('N°agents', fontsize=11)
        plt.xlabel('stake', fontsize=11)
        plt.hist(self.history.query(f"simulation == {0}").query(f"epoch == {0}")["stake"], bins=10)
        plt.subplot(1, 2, 2)
        plt.grid(axis='y')
        plt.legend()
        plt.title("Last epoch", fontsize=14)
        plt.xlabel('stake', fontsize=11)

        last_epoch = self.history['epoch'].max()
        # Media per ogni agente e std ?

        plt.hist(self.history.query(f"simulation == {0}").query(f"epoch == {last_epoch}")["stake"], bins=10)
        return fig

    def __rewards_gini_index(self):
        initial_stakes = self.history.query("epoch == 0").query("simulation == 0")
        history = self.history.merge(initial_stakes, on='id', suffixes=('', '_'))\
            .eval("stake = stake - stake_")\
            .drop(columns=['stake_'])

        history = history.drop(columns=['id']) \
            .groupby(['epoch', 'simulation'])['stake'] \
            .apply(gini_concentration_index) \
            .reset_index(name='gini') \
            .groupby('epoch') \
            .agg({'gini': ['mean', 'std']}) \
            .reset_index()

        fig = plt.figure()
        x = history['epoch']
        y_mean = history['gini']['mean']
        y_std_upper = y_mean + history['gini']['std']
        y_std_lower = y_mean - history['gini']['std']
        plt.plot(x, y_mean, label="mean ± std.dev", color='blue', linestyle="-")
        plt.plot(x, y_std_upper, color='lightsteelblue', linestyle="-.", linewidth=0.5, alpha=0.1)
        plt.plot(x, y_std_lower, color='lightsteelblue', linestyle="-.", linewidth=0.5, alpha=0.1)
        plt.fill_between(x=x.values, y1=y_std_upper.values, y2=y_std_lower.values, alpha=.1, color="blue")
        plt.grid()
        plt.legend()
        plt.title("Reward distributions", fontsize=17)
        plt.ylabel("Gini concentration index", fontsize=14)
        plt.xlabel("Time [epochs]", fontsize=14)
        return fig

    def __rewards_gini_ratio(self):
        initial_stakes = self.history.query("epoch == 0").query("simulation == 0")
        history = self.history.merge(initial_stakes, on='id', suffixes=('', '_'))\
            .eval("stake = stake - stake_")\
            .drop(columns=['stake_'])

        history = history.drop(columns=['id']) \
            .groupby(['epoch', 'simulation'])['stake'] \
            .apply(gini_concentration_ratio) \
            .reset_index(name='gini') \
            .groupby('epoch') \
            .agg({'gini': ['mean', 'std']}) \
            .reset_index()

        fig = plt.figure()
        x = history['epoch']
        y_mean = history['gini']['mean']
        y_std_upper = y_mean + history['gini']['std']
        y_std_lower = y_mean - history['gini']['std']
        plt.plot(x, y_mean, label="mean ± std.dev", color='blue', linestyle="-")
        plt.plot(x, y_std_upper, color='lightsteelblue', linestyle="-.", linewidth=0.5, alpha=0.1)
        plt.plot(x, y_std_lower, color='lightsteelblue', linestyle="-.", linewidth=0.5, alpha=0.1)
        plt.fill_between(x=x.values, y1=y_std_upper.values, y2=y_std_lower.values, alpha=.1, color="blue")
        plt.grid()
        plt.legend()
        plt.title("Reward distributions", fontsize=17)
        plt.ylabel("Gini concentration ratio", fontsize=14)
        plt.xlabel("Time [epochs]", fontsize=14)
        return fig

    def __stakes_for_agents(self):
        # TODO: std deviation of this ?
        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.grid(axis='y')
        plt.legend()
        plt.title("First epoch", fontsize=14)
        plt.ylabel('stakes', fontsize=11)
        plt.xlabel('agent id', fontsize=11)
        first_epoch_ordered_by_id = self.history.query("epoch == 0 and simulation == 0").sort_values(by=['id'])
        plt.bar(
            first_epoch_ordered_by_id['id'].astype(str),
            first_epoch_ordered_by_id['stake'],
            color='maroon',
            width=0.4
        )
        plt.subplot(1, 2, 2)
        plt.grid(axis='y')
        plt.legend()
        plt.title("Last epoch", fontsize=14)
        plt.xlabel('agent id', fontsize=11)

        last_epoch = self.history['epoch'].max()
        # Media per ogni agente e std ?
        last_epoch_ordered_by_id = self.history.query(f"epoch == {last_epoch} and simulation == 0").sort_values(by=['id'])
        plt.bar(last_epoch_ordered_by_id['id'].astype(str),
                last_epoch_ordered_by_id['stake'],
                color='maroon',
                width=0.4
                )
        return fig

    def __mean_and_std_stakes(self):
        """
        Calculate std of each epoch, then get the mean of each epoch std by simulations.
        """
        epochs_std = self.history.groupby(['simulation', 'epoch'])['stake'] \
            .std() \
            .groupby(['epoch']) \
            .mean()\
            .reset_index()
        epochs_mean = self.history.query('simulation == 0').groupby(['epoch'])['stake'] \
            .mean()\
            .reset_index()
        fig = plt.figure()
        x = epochs_std['epoch']
        y_mean = epochs_mean['stake']
        y_std_upper = y_mean + epochs_std['stake']
        y_std_lower = y_mean - epochs_std['stake']
        plt.plot(x, y_mean, label="mean ± std.dev", color='blue', linestyle="-")
        plt.plot(x, y_std_upper, color='lightsteelblue', linestyle="-.", linewidth=0.5, alpha=0.1)
        plt.plot(x, y_std_lower, color='lightsteelblue', linestyle="-.", linewidth=0.5, alpha=0.1)
        plt.fill_between(x=x.values, y1=y_std_upper.values, y2=y_std_lower.values, alpha=.1, color="blue")
        plt.grid()
        plt.legend()
        plt.title("Stake mean and std", fontsize=17)
        plt.ylabel("Mean", fontsize=14)
        plt.xlabel("Time [epochs]", fontsize=14)
        return fig

def stake_histogram_normalized(history):
    max_epochs = history[0]["epoch"].max()
    normalized_stakes_first = history[0].query(f"epoch == {0}")["stake"] / history[0].query(f"epoch == {0}")[
        "stake"].sum()
    normalized_stakes_last = history[0].query(f"epoch == {max_epochs}")["stake"] / \
                             history[0].query(f"epoch == {max_epochs}")["stake"].sum()

    plt.clf()
    plt.subplot(1, 2, 1)
    plt.grid(axis='y')
    plt.legend()
    plt.title("First epoch", fontsize=14)
    plt.ylabel('N°agents %', fontsize=11)
    plt.xlabel('stakes %', fontsize=11)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(10))
    plt.hist(normalized_stakes_first)

    plt.subplot(1, 2, 2)
    plt.grid(axis='y')
    plt.legend()
    plt.title("Last epoch", fontsize=14)
    plt.xlabel('stakes %', fontsize=11)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(10))
    plt.hist(normalized_stakes_last)
    plt.savefig('normalized_first_last_epochs_histogram.png')


def stake_histogram_evolution(history):
    data = history[0].groupby('epoch')["stake"].apply(list).to_list()
    max_epochs = history[0]["epoch"].max()
    columns = tuple(range(0, max_epochs + 1))
    rows = ['%d year' % x for x in (100, 50, 20, 10, 5)]

    values = np.arange(0, 2500, 500)
    value_increment = 1000

    # Get some pastel shades for the colors
    colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
    n_rows = len(data)

    index = np.arange(len(columns)) + 0.3
    bar_width = 0.4

    # Initialize the vertical-offset for the stacked bar chart.
    y_offset = np.zeros(len(columns))

    # Plot bars and create text labels for the table
    cell_text = []
    for row in range(n_rows):
        plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
        y_offset = y_offset + data[row]
        cell_text.append(['%1.1f' % (x / 1000.0) for x in y_offset])
    # Reverse colors and text labels to display the last value at the top.
    colors = colors[::-1]
    cell_text.reverse()

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.2)

    plt.ylabel("Loss in ${0}'s".format(value_increment))
    plt.yticks(values * value_increment, ['%d' % val for val in values])
    plt.title('Loss by Disaster')

    plt.show()


def time_series_histogram(history):
    h = history[0].groupby('epoch')["stake"].apply(np.histogram)

    a = np.random.random((16, 16))
    r = np.array(history[0].groupby(['epoch'], sort=False).stake.apply(list).tolist())

    plt.imshow(r[:10], cmap='hot', interpolation='nearest')
    plt.show()


def position_indexes(history):
    """ Using mean as position index """
    logging.info("Begin position_indexes")
    compact_history = [history[i].assign(simulation=i) for i in range(len(history))]
    compact_history = pd.concat(compact_history, ignore_index=True)
    compact_history = compact_history.drop(columns=['id'])
    simulations_describe_mean = compact_history.groupby(['epoch', 'simulation']).mean().reset_index().query(
        "simulation == 0")

    plt.clf()
    x = simulations_describe_mean["epoch"]
    y_mean = simulations_describe_mean["stake"]
    plt.plot(x, y_mean, label="mean ± std", color='blue', linestyle="-")
    plt.grid()
    plt.legend()
    plt.title("Stakes mean", fontsize=17)
    plt.ylabel("stakes mean", fontsize=14)
    plt.xlabel("time [epochs]", fontsize=14)
    plt.savefig('stakes_mean.png')
    logging.info("End position_indexes")


def dispersions_indexes(history):
    logging.info("Begin dispersion_indexes")
    compact_history = [history[i].assign(simulation=i) for i in range(len(history))]
    compact_history = pd.concat(compact_history, ignore_index=True)
    compact_history = compact_history.drop(columns=['id'])
    simulations_describe_std = compact_history.groupby(['epoch', 'simulation']).std().reset_index().groupby(
        'epoch').agg({'stake': ['mean', 'std']}).reset_index()
    simulations_describe_var = compact_history.groupby(['epoch', 'simulation']).var().reset_index().groupby(
        'epoch').agg({'stake': ['mean', 'std']}).reset_index()

    plt.clf()
    x = simulations_describe_std["epoch"]
    y_mean = simulations_describe_std["stake"]["mean"]
    std = simulations_describe_std["stake"]["std"]
    y_std_upper = y_mean + std
    y_std_lower = y_mean - std
    plt.plot(x, y_mean, label="mean ± std", color='blue', linestyle="-")
    plt.plot(x, y_std_upper, color='lightsteelblue', linestyle="-.", linewidth=0.5, alpha=0.1)
    plt.plot(x, y_std_lower, color='lightsteelblue', linestyle="-.", linewidth=0.5, alpha=0.1)
    plt.fill_between(x=x.values, y1=y_std_upper.values, y2=y_std_lower.values, alpha=.1, color="blue")
    plt.grid()
    plt.legend()
    plt.title("Stakes standard deviation", fontsize=17)
    plt.ylabel("std", fontsize=14)
    plt.xlabel("time [epochs]", fontsize=14)
    plt.savefig('stakes_std.png')

    plt.clf()
    x = simulations_describe_var["epoch"]
    y_mean = simulations_describe_var["stake"]["mean"]
    std = simulations_describe_var["stake"]["std"]
    y_std_upper = y_mean + std
    y_std_lower = y_mean - std
    plt.plot(x, y_mean, label="mean ± std", color='blue', linestyle="-")
    plt.plot(x, y_std_upper, color='lightsteelblue', linestyle="-.", linewidth=0.5, alpha=0.1)
    plt.plot(x, y_std_lower, color='lightsteelblue', linestyle="-.", linewidth=0.5, alpha=0.1)
    plt.fill_between(x=x.values, y1=y_std_upper.values, y2=y_std_lower.values, alpha=.1, color="blue")
    plt.grid()
    plt.legend()
    plt.title("Stakes variance", fontsize=17)
    plt.ylabel("var", fontsize=14)
    plt.xlabel("time [epochs]", fontsize=14)
    plt.savefig('stakes_var.png')
    logging.info("End dispersion_indexes")
