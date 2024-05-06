from numpy import ndarray
from dataclasses import dataclass, field
from collections import deque


Observation = ndarray
Observations = deque[Observation]


@dataclass(order=True)
class Neighbour:
    distance: float
    observation: Observation = field(compare=False)
