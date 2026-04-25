"""Solution controls"""


class Solver:
    """Defines a generic solver with options for iterative and multi-threaded
    solutions"""

    _n_threads: int
    _max_iterations: int
    _tol: float
    _alpha: float
    _edge: bool

    @property
    def n_threads(self):
        """Number of threads for solving."""
        return self._n_threads

    @property
    def max_iterations(self):
        """Maximum number of iterations for iterative solves."""
        return self._max_iterations

    @property
    def tol(self):
        """Convergence tolerance for iterative solves.

        !!! note
            This is typically a value of the H-field for magnetization calculations;
            the default value is roughly 1.3e-6 T, which may be more than necessary
            for most applications.
        """
        return self._tol

    @property
    def alpha(self):
        """Under-relaxation factor for iterative solves."""
        return self._alpha

    @property
    def edge(self):
        """Whether or not the edge-based element calculation should be used"""
        return self._edge


class DirectSolver(Solver):
    """Controls solver options for using the direct (full integration)
    solution routines"""

    def __init__(
        self, n_threads: int = 0, max_iterations=100, tol=1.0, alpha=0.5, edge=False
    ):
        self._n_threads = n_threads
        self._max_iterations = max_iterations
        self._tol = tol
        self._alpha = alpha
        self._edge = edge


class OctreeSolver(Solver):
    """Controls solver options for using the octree (Barnes-Hut) solution routines"""

    _theta: float
    _leaf_threshold: int

    def __init__(
        self,
        theta: float = 0.25,
        leaf_threshold: int = 16,
        n_threads: int = 0,
        max_iterations=100,
        tol=1.0,
        alpha=0.5,
        edge=False,
    ):
        self._theta = theta
        self._leaf_threshold = leaf_threshold
        self._n_threads = n_threads
        self._max_iterations = max_iterations
        self._tol = tol
        self._alpha = alpha
        self._edge = False

    @property
    def theta(self):
        """Returns the Barnes-Hut angle-opening criteria (accuracy control)"""
        return self._theta

    @property
    def leaf_threshold(self):
        """Returns the number of sources that will be evaluated individually at the
        octree leaf level"""
        return self._leaf_threshold
