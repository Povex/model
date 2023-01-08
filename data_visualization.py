import logging
from typing import List

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

from simulator.model.statistics.statistics import gini_irc, lorenz_curve


def visualization_gini_rcg(history):
    ginis = [simulation_df.groupby(['epoch'])["stake"].apply(gini_irc).to_frame().reset_index() for simulation_df in
             history]
    ginis_summary = pd.concat(ginis, ignore_index=True).groupby(["epoch"]).agg({'stake': ['mean', 'std']}).reset_index()
    plt.clf()
    x = ginis_summary["epoch"]
    y_mean = ginis_summary["stake"]["mean"]
    y_std_upper = y_mean + ginis_summary["stake"]["std"]
    y_std_lower = y_mean - ginis_summary["stake"]["std"]
    plt.plot(x, y_mean, label="mean ± std.dev", color='blue', linestyle="-")
    plt.plot(x, y_std_upper, color='lightsteelblue', linestyle="-.", linewidth=0.5, alpha=0.1)
    plt.plot(x, y_std_lower, color='lightsteelblue', linestyle="-.", linewidth=0.5, alpha=0.1)
    plt.fill_between(x=x.values, y1=y_std_upper.values, y2=y_std_lower.values, alpha=.1, color="blue")
    plt.grid()
    plt.legend()
    plt.title("Stake distributions", fontsize=17)
    plt.ylabel("Gini coefficient", fontsize=14)
    plt.xlabel("Time [epochs]", fontsize=14)
    plt.savefig('simulation.png')

    plt.clf()
    first_gini = ginis[0].query("epoch == 0")["stake"].values[0]
    for gini_df in ginis:
        gini_df["stake"] -= first_gini
    ginis_summary = pd.concat(ginis, ignore_index=True).groupby(["epoch"]).agg({'stake': ['mean', 'std']}).reset_index()
    plt.clf()
    x = ginis_summary["epoch"]
    y_mean = ginis_summary["stake"]["mean"]
    y_std_upper = y_mean + ginis_summary["stake"]["std"]
    y_std_lower = y_mean - ginis_summary["stake"]["std"]
    plt.plot(x, y_mean, label="mean ± std.dev", color='blue', linestyle="-")
    plt.plot(x, y_std_upper, color='lightsteelblue', linestyle="-.", linewidth=0.5, alpha=0.1)
    plt.plot(x, y_std_lower, color='lightsteelblue', linestyle="-.", linewidth=0.5, alpha=0.1)
    plt.fill_between(x=x.values, y1=y_std_upper.values, y2=y_std_lower.values, alpha=.1, color="blue")
    plt.grid()
    plt.legend()
    plt.title("Simulation", fontsize=17)
    plt.ylabel("Differences trend", fontsize=14)
    plt.xlabel("Time [epochs]", fontsize=14)
    plt.savefig('simulation_diff.png')


def gini_on_stake_rewards(history: List[pd.DataFrame]):
    initial_stakes = history[0].query("epoch == 0").drop(columns=['epoch'])
    history_rewards = [simulation_df.merge(initial_stakes, on='id', suffixes=('', '_')).eval("stake = stake - stake_")
                       .drop(columns=['stake_']) for simulation_df in history]
    ginis = [simulation_df.groupby(['epoch'])["stake"].apply(gini_irc).to_frame().reset_index() for simulation_df in
             history_rewards]
    ginis_summary = pd.concat(ginis, ignore_index=True).groupby(["epoch"]).agg({'stake': ['mean', 'std']}).reset_index()
    plt.clf()
    x = ginis_summary["epoch"]
    y_mean = ginis_summary["stake"]["mean"]
    y_std_upper = y_mean + ginis_summary["stake"]["std"]
    y_std_lower = y_mean - ginis_summary["stake"]["std"]
    plt.plot(x, y_mean, label="mean ± std.dev", color='blue', linestyle="-")
    plt.plot(x, y_std_upper, color='lightsteelblue', linestyle="-.", linewidth=0.5, alpha=0.1)
    plt.plot(x, y_std_lower, color='lightsteelblue', linestyle="-.", linewidth=0.5, alpha=0.1)
    plt.fill_between(x=x.values, y1=y_std_upper.values, y2=y_std_lower.values, alpha=.1, color="blue")
    plt.grid()
    plt.legend()
    plt.title("Reward distributions", fontsize=17)
    plt.ylabel("Gini coefficient", fontsize=14)
    plt.xlabel("Time [epochs]", fontsize=14)
    plt.savefig('simulation_gini_rewards.png')


def stake_histogram(history):
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.grid(axis='y')
    plt.legend()
    plt.title("First epoch", fontsize=14)
    plt.ylabel('N°agents', fontsize=11)
    plt.xlabel('stake', fontsize=11)
    plt.hist(history[0].query(f"epoch == {0}")["stake"], bins=10)

    plt.subplot(1, 2, 2)
    plt.grid(axis='y')
    plt.legend()
    plt.title("Last epoch", fontsize=14)
    plt.xlabel('stake', fontsize=11)
    plt.hist(history[0].query(f"epoch == {1000}")["stake"], bins=10)
    plt.savefig('simulation_first_last_epochs_histogram.png')


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


def lorenz_curves_3d(history):
    # TODO: aggregate simulation stakes for lorenz curves
    compact_history = [history[i].assign(simulation=i) for i in range(len(history))]
    compact_history = pd.concat(compact_history, ignore_index=True)
    compact_history = compact_history.drop(columns=['id'])

    last_epoch = compact_history["epoch"].max()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    def plot2din3d(x, y, z):
        ax.plot(x, y, zs=z, color='royalblue')
        ax.plot(x, x, z, color='black', alpha=.1)

    n_planes = 10
    steps = int((last_epoch + 1) / n_planes)
    for z in range(0, last_epoch + 1, steps):
        f, q = lorenz_curve(compact_history.query(f"epoch == {z}").query("simulation == 0")["stake"])
        plot2din3d(f, q, z)

    # ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(last_epoch, 0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Epochs')

    ax.view_init(elev=110., azim=-72.0)

    plt.show()
