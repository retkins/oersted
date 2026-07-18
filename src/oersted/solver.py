"""Solution controls"""

from dataclasses import dataclass
from typing import Literal, get_args

Method = Literal["direct", "octree"]
Integration = Literal["element", "point"]
MultipoleOrder = Literal["monopole", "dipole"]


@dataclass(frozen=True, slots=True, kw_only=True)
class SolverSettings:
    """Define settings to use during a solve

    Attributes:
        method: use either "direct" O(N^2) or "octree" O(N log(N)) integration for
            Biot-Savart law integration
        integration: use either "element" for full integration (high
            accuracy, slow) or "point" for a point-source approximation (low accuracy
            near the source, but ~40x faster)
        n_threads: number of cpu threads to use for solution; default is 0 for all
            available
        theta: the Barnes-Hut angle-opening criteria (theta >= 0.0)
        near_field_ratio: (deprecated) ratio of target distance to source element size;
            defines the 'mid-field' during a barnes-hut solve (alpha >= 0.0)
        max_leaf_size: defines the maximum leaf size before splitting in the octree
            (>=1)
        batch_size: defines the number of target points to process at a time (for
            Barnes-Hut solves))
        multipole_order: defines the multipole expansion order for Barnes-Hut solves;
            currently supported are "monopole" and "dipole"
        max_iterations: maximum number of iterations before solution returns early
        atol: absolute tolerance for convergence criteria
        under_relaxation_factor: provides solution stability for fixed-point iteration
    """

    # General settings
    method: Method = "direct"
    integration: Integration = "element"
    n_threads: int = 0

    # Octree settings
    theta: float = 0.5
    near_field_ratio: float = 10.0
    max_leaf_size: int = 16
    batch_size: int = 1
    multipole_order: MultipoleOrder = "dipole"

    # Iterative solve settings
    max_iterations: int = 100
    atol: float = 1e-6
    under_relaxation_factor: float = 0.5

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
        if self.theta < 0.0:
            raise ValueError(f"theta must be > 0, got {self.theta}")
        if self.near_field_ratio < 0.0:
            raise ValueError(f"alpha must be > 0, got {self.near_field_ratio}")
        if self.max_leaf_size < 1:
            raise ValueError(f"Max leaf size must be >= 1, got {self.max_leaf_size}")
        if self.max_iterations < 1:
            raise ValueError(
                f"Max solver iterations must be >= 1, got `{self.max_iterations}"
            )
        if self.atol < 0:
            raise ValueError(f"Relative error tolerance must be > 0, got `{self.atol}")
        if self.under_relaxation_factor > 1.0 or self.under_relaxation_factor < 0.0:
            raise ValueError(
                f"Under relaxation factor must be in range [0,1], got \
                    `{self.under_relaxation_factor}"
            )


DEFAULT_SETTINGS = SolverSettings()
