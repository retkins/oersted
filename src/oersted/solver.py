class Solver:
    _n_threads: int

    @property
    def n_threads(self):
        return self._n_threads


class DirectSolver(Solver):
    def __init__(self, n_threads: int = 0):
        self._n_threads = n_threads


class OctreeSolver(Solver):
    _theta: float
    _leaf_threshold: int

    def __init__(self, theta: float = 0.25, leaf_threshold: int = 16, n_threads: int = 0):

        self._theta = theta
        self._leaf_threshold = leaf_threshold
        self._n_threads = n_threads

    @property
    def theta(self):
        return self._theta

    @property
    def leaf_threshold(self):
        return self._leaf_threshold
