from typeguard import typechecked

from .distributions import Distribution


@typechecked
class Prior:
    """A prior probability distribution

    Attributes:
        name (str): This is the name given to the prior, usually corresponding to the parameter it represents.
        distribution (Distribution): The one-dimensional probability distribution for the prior.

    """

    def __init__(self, name: str, distribution: Distribution):
        self.name = name
        self.distribution = distribution

    def to_json(self):
        return {"name": self.name, "distribution": self.distribution.to_json()}
