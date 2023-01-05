from typing import List

import numpy as np
import pandas as pd


def gini(x):
    if type(x) != np.array: x = np.array(x)
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x) ** 2 * np.mean(x))


def gini_irc(x):
    """ Indice relativo di concentrazione di Gini, mi sembra più accurato"""
    if type(x) == pd.DataFrame or type(x) == pd.Series:
        x = x.values.tolist()
    volume = sum(x)
    if volume == 0: return 0
    x.sort()
    l = len(x)
    q = []
    p = []
    tmp = 0
    g = 0
    for i in range(len(x)):
        tmp += x[i]
        q.append(tmp / volume)
        p.append((i + 1) / l)
        g += (p[i] - q[i])
    return g * (2 / (l - 1))


def gini_concentration_index(x):
    """ x must be a numpy array or a pd.DataFrame sliced on a column (aka pd.Series)
        https://dariomalchiodi.gitlab.io/sad-python-book/L05-Indici_di_eterogeneit%C3%A0.html
     """
    q = np.sort(x).cumsum() / np.sum(x)
    n = len(x)
    f = np.arange(1, n + 1) / float(n)
    return 2 * np.sum(f - q) / (n - 1)


def gini_concentration_ratio(x):
    n = len(x)
    return gini_concentration_index(x) * (n - 1) / n


def R(x: List[float]):
    x.sort()
    volume = sum(x)
    l = len(x)
    q = []
    p = []
    tmp = 0
    g = 0
    for i in range(len(x)):
        tmp += x[i]
        q.append(tmp / volume)
        p.append((i + 1) / l)
        g += (p[i] - q[i])
    return g


def gini_rcg(x: List[float]):
    """ Rapporto di concentrazione di Gini """
    x.sort()
    volume = sum(x)
    l = len(x)
    q = []
    p = []
    tmp = 0
    g = 0
    for i in range(len(x)):
        tmp += x[i]
        q.append(tmp / volume)
        p.append((i + 1) / l)
        g += (p[i] - q[i])
    return g * (l - 1) / l


def lorenz_curve(x: List[float]):
    x = sorted(x)
    volume = sum(x)
    tmp = 0
    q = []
    for i in range(len(x)):
        tmp += x[i]
        q.append(tmp / volume)
    return q
