import typing as tp
import numpy as np

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

    def start(self) -> None:
        self._knngraph.build()

    def update(self, observation: Observation) -> None:
        self._knngraph.update(observation)

    def calculate_random_variable(self, permutation: np.array, t: int):
        def b(i: int, j: int) -> bool:
            pi = permutation[i]
            pj = permutation[j]
            return (pi <= t < pj) or (pj <= t < pi)

        s = 0

        for i in range(self._window_size):
            for j in range(self._window_size):
                s += (self._knngraph.check_neighbour(i, j) + self._knngraph.check_neighbour(i, j)) * b(i, j)

        return s

    def calculate_statistics(self):
        permutation: np.array = np.arange(self._window_size)
        np.random.shuffle(permutation)



