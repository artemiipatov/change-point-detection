import typing as tp
from collections import deque
from itertools import islice

from src.utils.observation import Observation, Observations
from src.utils.observation_heap import NNHeap


class KNNGraph:
    def __init__(self, observations_count: int,  k=3,  observations: Observations = None,
                 metric: tp.Callable[[Observation, Observation], float] = None):
        self._k = k
        self._observations_count = observations_count
        self._observations = observations
        self._window = deque(islice(self._observations,
                                  len(self._observations) - self._observations_count - 1,
                                  len(self._observations)))
        self._metric = metric
        self._graph: deque[NNHeap] | None = None

    def build(self) -> None:
        assert self._graph is None

        self._graph = deque(maxlen=self._observations_count)

        for i in range(0, self._observations_count):
            heap = NNHeap(self._k, self._metric, self._observations[-i - 1])
            heap.build(self._window)
            self._graph.appendleft(heap)

    def update(self, observation: Observation) -> None:
        obsolete_obs = self._window[0]
        self._observations.append(observation)
        self._window.append(observation)
        self._graph.popleft()

        for heap in self._graph:
            heap.remove(obsolete_obs, self._window)
            heap.add(observation)

        new_heap = NNHeap(self._k, self._metric, observation)
        new_heap.build(self._window)
        self._graph.append(new_heap)
