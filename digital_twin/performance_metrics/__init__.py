from .fd import frechet_distance
from .mmd import mmd_rbf
from .misc import (
    absolute_percentage_error as ape,
    mean_estimation_absolute_percentage_error as meape,
    std_estimation_absolute_percentage_error as seape,
    aggregate_loads,
)

# TODO: whats the point of aggregate_loads
__all__ = ["frechet_distance", "mmd_rbf", "ape", "meape", "seape", "aggregate_loads"]
