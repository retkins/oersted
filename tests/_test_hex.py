from oersted import bfield_hexahedron, b_field, CentroidMesh
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

nodes = np.array(
    [
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
        [-0.5, -0.5, 0.5],
    ]
)
jdensity = np.array([0.0, -1e6, 0.0])

n = 100
d = np.linspace(0.5, 4, n)
targets = np.zeros((n, 3))
targets[:, 0] = d / np.sqrt(2)
targets[:, 1] = d / np.sqrt(2)

_b = np.zeros((n, 3))

centroids = np.array([[0.0, 0.0, 0.0]])
vol = np.array([1.0])

mesh = CentroidMesh(centroids, vol)

_bpoint = b_field(mesh, jdensity, targets)

for i in range(0, n):
    _b[i, :] = bfield_hexahedron(nodes, jdensity, targets[i, :])

b = norm(_b, axis=1)
bpoint = norm(_bpoint, axis=1)

plt.plot(1.0 / targets[:, 0], bpoint / b, label="hex")
# plt.plot(targets[:,0], bpoint, label='point')
# plt.legend()
plt.xlabel("theta = element size / distance from element centroid")
plt.ylabel("Ratio of field (point/hex)")
plt.title("line along diagonal, (d, d, 0)/sqrt(2)")
