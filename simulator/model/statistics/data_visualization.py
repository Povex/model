import logging

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import scipy.stats as stats
import math

import io

from tqdm import tqdm

from src.model.model_config import ModelConfig


def summary_statistics(distribution, epoch, simulation):
    dataset = pd.DataFrame({"stake": distribution})
    print(f"[{epoch}][{simulation}]", dataset.describe())


def normal(distribution, epoch, simulation):
    mu = stats.tmean(distribution)
    variance = stats.variation(distribution)
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma))
    plt.savefig(f"res/{simulation}_{epoch}_normal.png")
    plt.clf()


def hist(distribution, epoch, simulation):
    plt.hist(distribution)
    plt.savefig(f"res/{simulation}_{epoch}_hist.png")
    plt.clf()


def hist_bins(distribution, epoch, simulation):
    plt.hist(distribution, bins=range(max(distribution) + 1))
    plt.savefig(f"res/{simulation}_{epoch}_hist_bins.png")
    plt.clf()


def pie_chart(history, epoch, simulation):
    history = history[epoch].sort_values(by=['stake'])
    stakes = history["stake"]
    agent_ids = history["id"]
    plt.pie(stakes, labels=agent_ids)
    plt.savefig(f"res/{simulation}_{epoch}_pie.png")
    plt.clf()


def data_analysis(history):
    logging.info("Begin data analysis section")
    for epoch in [0, len(history[0]) - 1]:
        summary_statistics(history[0][epoch]["stake"], epoch, 1)
        hist(history[0][epoch]["stake"], epoch, 1)
        pie_chart(history[1], epoch, 1)
        normal(history[0][epoch]["stake"], epoch, 1)
    for epoch in [0, len(history[0]) - 1]:
        summary_statistics(history[1][epoch]["stake"], epoch, 2)
        hist(history[1][epoch]["stake"], epoch, 2)
        pie_chart(history[0], epoch, 2)
        normal(history[1][epoch]["stake"], epoch, 2)
    logging.info("End data analysis section")


def data_visualization(stakes, initial_or_final):
    # Continuous Probability Distributions
    # Range of values
    # Easy to analyze with Gaussian
    # altre.. (Weibull distribution and the lognormal distribution)
    agent_ids = [i[0] for i in stakes]
    agent_stakes = [round(i[1]) for i in stakes]

    dataset = pd.DataFrame({"stake": agent_stakes})
    print(f"[{initial_or_final}] ", dataset.describe())

    mu = stats.tmean(agent_stakes)
    variance = stats.variation(agent_stakes)
    sigma = math.sqrt(variance)

    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma))

    plt.clf()

    counts, bins = np.histogram(agent_stakes, density=True)
    plt.stairs(counts, bins)
    plt.savefig(f"res/{initial_or_final}_prob_func.png")
    plt.clf()

    plt.hist(agent_stakes)
    plt.savefig(f"res/{initial_or_final}_hist.png")

    max_value = max(agent_stakes)
    max_index = agent_stakes.index(max_value)
    richest_agent = agent_ids[max_index]

    plt.pie(agent_stakes, labels=agent_ids)
    plt.savefig(f"res/{initial_or_final}_pie.png")
    plt.clf()


def mean_simulations(history, epoch):
    df = pd.DataFrame()
    plt.clf()
    for simulation in range(len(history)):
        if simulation == 0:
            df = history[0][epoch]
        else:
            df = df.append(history[simulation][epoch])
    df = df.groupby(['id']).mean()
    return df


def hist_plot(df: pd.DataFrame, epoch: int):
    plt.hist(df['stake'])
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf


class DataVisualization:
    pass
