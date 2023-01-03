from matplotlib import pyplot as plt
import pandas as pd
from simulator.model.statistics.statistics import gini_irc


def visualization_gini(history):
    print("Generating gini plot, please wait..")
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
    print("Generating trend differences gini plot, please wait..")
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
