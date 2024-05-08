from numpy import ndarray
from dataclasses import dataclass, field
from collections import deque
from typing import TypeAlias


Observation: TypeAlias = ndarray
Observations: TypeAlias = deque[Observation]


@dataclass(order=True)
class Neighbour:
    distance: float
    observation: Observation = field(compare=False)
