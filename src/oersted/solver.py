"""Solution controls"""

from dataclasses import dataclass, field
from typing import Literal, get_args

Method = Literal["direct", "octree"]
Integration = Literal["element", "point"]


@dataclass(frozen=True, slots=True, kw_only=True)
class OctreeSettings:
    """Defines settings for solutions that use an octree/barnes-hut solver

    Attributes:
        theta: the Barnes-Hut angle-opening criteria (theta >= 0.0)
        near_field_ratio: ratio of target distance to source element size; defines the
            'mid-field' during a barnes-hut solve (alpha >= 0.0)
        max_leaf_size: defines the maximum leaf size before splitting in the octree
            (>=1)
    """

    theta: float = 0.5
    near_field_ratio: float = 10.0
    max_leaf_size: int = 16

    def __post_init__(self):
        if self.theta < 0.0:
            raise ValueError(f"theta must be > 0, got {self.theta}")
        if self.near_field_ratio < 0.0:
            raise ValueError(f"alpha must be > 0, got {self.near_field_ratio}")
        if self.max_leaf_size < 1:
            raise ValueError(f"Max leaf size must be >= 1, got {self.max_leaf_size}")


@dataclass(frozen=True, slots=True, kw_only=True)
class IterationSettings:
    """Defines solver settings for iterative solutions

    Attributes:
        max_iterations: maximum number of iterations before solution returns early
        rtol: relative tolerance for convergence criteria
        under_relaxation_factor: provides solution stability for fixed-point iteration
    """

    max_iterations: int = 100
    rtol: float = 1e-6
    under_relaxation_factor: float = 0.5

    def __post_init__(self):
        if self.max_iterations < 1:
            raise ValueError(
                f"Max solver iterations must be >= 1, got `{self.max_iterations}"
            )
        if self.rtol < 0:
            raise ValueError(f"Relative error tolerance must be > 0, got `{self.rtol}")
        if self.under_relaxation_factor > 1.0 or self.under_relaxation_factor < 0.0:
            raise ValueError(
                f"Under relaxation factor must be in range [0,1], got \
                    `{self.under_relaxation_factor}"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class SolverSettings:
    """Define settings to use during a solve

    Attributes:
        method: use either "direct" O(N^2) or "octree" O(N log(N)) integration for
            Biot-Savart law integration
        integration: use either "element" for full integration (high
            accuracy, slow) or "point" for a point-source approximation
        n_threads: number of cpu threads to use for solution; default is 0 for all
            available
        octree: settings for octree/barnes-hut solutions
        iteration: settings for iterative solves
    """

    method: Method = "direct"
    integration: Integration = "element"
    n_threads: int = 0
    octree: OctreeSettings = field(default_factory=OctreeSettings)
    iteration: IterationSettings = field(default_factory=IterationSettings)

    def __post_init__(self):
        if self.method not in get_args(Method):
            raise ValueError(
                f"Solution method must be one of {get_args(Method)}, \
                    got {self.method!r}"
            )
        if self.integration not in get_args(Integration):
            raise ValueError(
                f"Near-field integration method must be one of \
                    {get_args(Integration)}, got `{self.integration!r}"
            )
        if self.n_threads < 0:
            raise ValueError(
                f"Number of cpu threads must be greater than 1, got `{self.n_threads}`"
            )


DEFAULT_SETTINGS = SolverSettings()
