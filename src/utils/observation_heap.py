import heapq
import typing as tp

from observation import Observation, Neighbour, Observations


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

    def add(self, neighbour: Observation) -> None:
        if not self.try_add(self._heap, neighbour):
            _ = self.try_add(self._auxiliary_heap, neighbour)

    def try_add(self, heap: list[Neighbour], observation: Observation) -> bool:
        if observation is self._main_observation:
            return False

        neg_distance = -self._metric(self._main_observation, observation)

        if len(heap) < self._size or neg_distance > heap[0].distance:
            neighbour = Neighbour(neg_distance, observation)
            heapq.heapreplace(heap, neighbour)
            return True

        return False

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
