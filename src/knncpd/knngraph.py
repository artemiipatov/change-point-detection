import typing as tp
from collections import deque
from itertools import islice

from utils.observation import Observation, Observations
from utils.observation_heap import NNHeap


class KNNGraph:
    def __init__(self, observations_count: int, observations: Observations,
                 metric: tp.Callable[[Observation, Observation], float], k=3) -> None:
        self._k = k
        self._observations_count = observations_count
        self._observations = observations
        self._window = deque(islice(self._observations,
                                    len(self._observations) - self._observations_count,
                                    len(self._observations)))
        self._metric = metric
        self._graph: deque[NNHeap] = deque(maxlen=self._observations_count)

    def build(self) -> None:
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

    def check_neighbour(self, obs_index: int, neighbour_index: int) -> bool:
        neighbour = self._window[neighbour_index]
        return self._graph[obs_index].find_in_heap(neighbour)
