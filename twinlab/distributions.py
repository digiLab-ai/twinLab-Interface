from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from typeguard import typechecked


@typechecked
class Distribution(ABC):
    @abstractmethod
    def to_json(self) -> np.array:
        pass


@typechecked
class DistributionMethods(Enum):
    UNIFORM = "uniform"


@typechecked
class Uniform(Distribution):
    """A one-dimensional continous uniform distribution between a minimum and maximum value.

    Args:
        min (float): The minimum value of the distribution.
        max (float): The maximum value of the distribution.

    """

    def __init__(self, min: float, max: float):
        self.min = min
        self.max = max

    def to_json(self):
        return {
            "method": DistributionMethods.UNIFORM.value,
            "distribution_params": {"max": self.max, "min": self.min},
        }
