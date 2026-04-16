"""Define material properties for magnetostatics calculations"""

from dataclasses import dataclass
from numpy import array
from numpy.typing import NDArray
from numpy import float64, interp
import oersted
from .constants import MU0


@dataclass
class BHCurve:
    """B-H curve for a nonlinear magnetic material"""

    h_values: NDArray[float64]
    b_values: NDArray[float64]

    def lookup(self, h: float) -> float:
        """Linearly interpolate from the B-H curve"""

        return float(interp(h, self.h_values, self.b_values))


class Material:
    def mu_r(self, h: float) -> float:
        raise NotImplementedError

    def chi(self, h: float) -> float:
        """Return the volume susceptibility of the material, defined by
        `chi = mu_r - 1.0`
        """
        return self.mu_r(h) - 1.0

    def to_mh_curve(self) -> tuple[NDArray[float64], NDArray[float64]]:
        raise NotImplementedError


@dataclass
class FreeSpace(Material):
    def mu_r(self, h: float) -> float:
        """Free space (vacuum) has a relative permeability of exactly 1.0"""
        return 1.0


@dataclass
class LinearMaterial(Material):
    """A linear magnetic material that has a constant mu_r for all
    values of applied H-field.
    """

    _mu_r: float

    def __init__(self, _mu_r):
        self._mu_r = _mu_r

    def mu_r(self, h: float):
        """Return the relative magnetic permeability"""
        return self._mu_r

    def to_mh_curve(self) -> tuple[NDArray[float64], NDArray[float64]]:
        """Convert the B-H curve to an M-H curve, which is more convenient for
        magnetization calculations"""
        h_values = array([0.0, 1e10])
        m_values = array([(self._mu_r - 1.0) * h for h in h_values])
        return (h_values, m_values)


@dataclass
class NonlinearMaterial(Material):
    """A nonlinear magnetic material that has a B-H curve"""

    curve: BHCurve

    def __init__(self, curve: BHCurve):
        self.curve = curve

    def mu_r(self, h: float) -> float:
        """Return the relative magnetic permeability at a given H-field strength"""
        b: float = self.curve.lookup(h)

        if h != 0.0:
            return (b / oersted.MU0) / h - 1.0
        else:
            return 1.0

    def to_mh_curve(self) -> tuple[NDArray[float64], NDArray[float64]]:
        """Convert the B-H curve to an M-H curve, which is more convenient for
        magnetization calculations"""
        return (self.curve.h_values, self.curve.b_values / MU0 - self.curve.h_values)
