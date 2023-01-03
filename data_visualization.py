from typing import List

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from simulator.model.statistics.statistics import gini_irc
from mpl_toolkits.mplot3d import Axes3D


def visualization_gini(history):
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
    plt.title("Simulation", fontsize=17)
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
    plt.title("Simulation", fontsize=17)
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


