import typing as tp

import knngraph
from src.utils.observation import Observation, Observations


class KNNCPD:
    def __init__(self, window_size: int, k=3, metric: tp.Callable[[Observation, Observation], float] = None,
                 threshold: float = 0.5, package_size: int = 1, observations: Observations = None, offset=0):
        self._k = k
        self._metric = metric
        self._threshold = threshold
        self._package_size = package_size
        self._observations = observations
        self._offset = offset
        self._window_size = window_size
        self._knngraph = knngraph.KNNGraph(k, observations, metric)

    def start(self):
        self._knngraph.build()

    def update(self, observation: Observation):
        self._knngraph.remove_last()
        self._knngraph.add(observation)
