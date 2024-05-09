import typing as tp
import numpy as np
from statistics import mean
from math import sqrt

import knngraph
from src.utils.observation import Observation, Observations


class KNNCPD:
    def __init__(self, window_size: int,  observations: Observations,
                 metric: tp.Callable[[Observation, Observation], float], k=3,
                 threshold: float = 0.5, package_size: int = 1, offset=0) -> None:
        self._k = k
        self._metric = metric
        self._threshold = threshold
        self._package_size = package_size
        self._observations = observations
        self._offset = offset
        self._window_size = window_size
        self._knngraph = knngraph.KNNGraph(k, observations, metric)
        self._statistics: float = 0.0

    def start(self) -> None:
        self._knngraph.build()

    def update(self, observation: Observation) -> None:
        self._knngraph.update(observation)
        self._statistics = self.calculate_statistics()

    def calculate_random_variable(self, permutation: np.array, t: int) -> int:
        def b(i: int, j: int) -> bool:
            pi = permutation[i]
            pj = permutation[j]
            return (pi <= t < pj) or (pj <= t < pi)

        s = 0

        for i in range(self._window_size):
            for j in range(self._window_size):
                s += (self._knngraph.check_neighbour(i, j) + self._knngraph.check_neighbour(i, j)) * b(i, j)

        return s

    def calculate_statistics(self) -> float:
        permutation: np.array = np.arange(self._window_size)
        np.random.shuffle(permutation)

        expectation = mean(self.calculate_random_variable(permutation, i) for i in range(self._window_size))
        expectation_sqr = mean(self.calculate_random_variable(permutation, i)**2 for i in range(self._window_size))
        deviation = sqrt(expectation_sqr - expectation**2)
        statistics = -(self.calculate_random_variable(permutation, self._window_size - 1) - expectation) / deviation

        return statistics

    def check_change_point(self) -> bool:
        return self._statistics > self._threshold
