import heapq
import typing as tp

from utils.observation import Observation, Neighbour, Observations


class NNHeap:
    def __init__(self, size: int, metric: tp.Callable[[Observation, Observation], float],
                 main_observation: Observation) -> None:
        self._size = size
        self._metric = metric
        self._main_observation = main_observation
        self._heap: list[Neighbour] = []
        self._auxiliary_heap: list[Neighbour] = []

    def build(self, neighbours: Observations) -> None:
        for neighbour in neighbours:
            self.add(neighbour)

    def add(self, observation: Observation) -> None:
        if observation is self._main_observation:
            return

        neg_distance = -self._metric(self._main_observation, observation)
        neighbour = Neighbour(neg_distance, observation)

        if len(self._heap) == self._size and neighbour.distance > self._heap[0].distance:
            old_neighbour = heapq.heapreplace(self._heap, neighbour)
            self.add_auxiliary(old_neighbour)
        elif len(self._heap) == self._size and neighbour.distance <= self._heap[0].distance:
            self.add_auxiliary(neighbour)
            heapq.heappush(self._heap, neighbour)
        else:
            heapq.heappush(self._heap, neighbour)

    def add_auxiliary(self, neighbour: Neighbour) -> None:
        if len(self._auxiliary_heap) == self._size and neighbour.distance > self._auxiliary_heap[0].distance:
            heapq.heapreplace(self._auxiliary_heap, neighbour)
        else:
            heapq.heappush(self._auxiliary_heap, neighbour)

    def remove(self, observation: Observation, observations: Observations) -> None:
        if not self._heap:
            return

        neg_distance = -self._metric(self._main_observation, observation)

        if neg_distance < self._heap[0].distance:
            return

        neighbour = Neighbour(neg_distance, observation)
        self._heap.remove(neighbour)

        if not self._auxiliary_heap:
            new_neighbour = heapq.heappop(self._auxiliary_heap)
            heapq.heappush(self._heap, new_neighbour)
            heapq.heapify(self._heap)
        else:
            self.build(observations)

    def find_in_heap(self, observation: Observation) -> bool:
        def predicate(x: Neighbour) -> bool:
            return x.observation is observation

        return any(predicate(i) for i in self._heap)
