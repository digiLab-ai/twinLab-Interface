from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from typeguard import typechecked


@typechecked
class SamplingMethods(Enum):
    LATIN_HYPERCUBE = "latin_hypercube"
    UNIFORM_RANDOM = "uniform_random"


@typechecked
class Sampling(ABC):
    @abstractmethod
    def to_json(cls, sampling_params: dict):
        pass


@typechecked
class LatinHypercube(Sampling):
    """A sampling strategy that uses Latin Hypercube Sampling.

    Args:
        scramble (bool, optional): Whether to scramble the samples within sub-cubes.
            The default value is ``True``.
        optimization (str | None, optional): The optimization method to use for generating the samples.
            Options are:

            - ``None``: No optimization is performed once the intial samples are generated.
            - ``"random-cd"``: Randomly permute the columns of the matrix in order to lower the centred discrepancy of the generated samples.
            - ``"lloyd"``: Perturb the samples using a modified Lloyd-Max algorithm.
              The process converges to equally spaced samples.

            The default is ``"random-cd"``.

    """

    def __init__(
        self, scramble: bool = True, optimization: Optional[str] = "random-cd"
    ):
        self.scramble = scramble
        self.optimization = optimization

    def to_json(self):
        return {
            "method": SamplingMethods.LATIN_HYPERCUBE.value,
            "sampling_params": {
                "scramble": self.scramble,
                "optimization": self.optimization,
            },
        }


@typechecked
class UniformRandom(Sampling):
    """A sampling strategy that random-uniformly samples the  space."""

    def __init__(self):
        pass

    def to_json(self):
        return {
            "method": SamplingMethods.UNIFORM_RANDOM.value,
            "sampling_params": {},
        }
