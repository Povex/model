from typing import List

import numpy as np


def gini(x):
    if type(x) != np.array: x = np.array(x)
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x) ** 2 * np.mean(x))


def gini_irc(x):
    """ Indice relativo di concentrazione di Gini, mi sembra più accurato"""
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
    return g * (2 / (l - 1))


def R(x: List[int]):
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


def gini_rcg(x: List[int]):
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
