from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from collections import deque

import knncpd.knncpd as cpd
import utils.observation as obs


def metric(obs1: obs.Observation, obs2: obs.Observation) -> float:
    return sqrt(sum((obs1[i] - obs2[i]) ** 2 for i in range(obs1.size)))


START_YEAR = 1851

series = pd.read_excel('../datasets/COAL MINING DISASTERS UK.xlsx')['Count'].to_numpy().astype(np.float64)
model = cpd.KNNCPD(15, metric)
statistics_list = []

for observation in series:
    model.update(np.array([observation]))
    statistics_list.append(model.statistics)
    if len(statistics_list) == 20:
        print(model._knngraph._window)

print(statistics_list)
#
# years = np.arange(START_YEAR, START_YEAR + series.size + 1, 1)
#
# plt.plot(years, statistics_list)
#
# plt.title('Статистика')
# plt.xlabel('Год')
# plt.ylabel('Статистика')
#
# plt.show()