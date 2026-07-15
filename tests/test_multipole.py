""" Confirm that the dipole correction works as intended

References:
https://arxiv.org/pdf/2211.08438
https://www.wtamu.edu/~cbaird/Lecture11.pdf
""" 
from typing import Literal
import numpy as np
import oersted

MU0_4PI = oersted.MU0 / (4.0 * np.pi)
mesh_size = 10e-3
jmag = 1e10
mesh, jdensity = oersted.make_helmholtz("tests/data/ring.stp", mesh_size,jmag=jmag)

# min xyz: [-0.20743792 -0.20743327 -0.10852091]
# max xyz: [ 0.20747637  0.20743934  0.10855099]
# print(f"min xyz: {np.min(mesh.centroids, axis=0)}")

mask = (mesh.centroids < 0).all(axis=1)

src_pts = mesh.centroids[mask, :]
src_vol = mesh.volumes[mask]
src_jdensity = jdensity[mask, :]


def bfield_all(src_pts, src_vol, src_jdensity, targets):
    b = np.zeros_like(targets)
    n = src_pts.shape[0]
    for i in range(n):
        rp = targets - src_pts[i, :]
        rmag = np.linalg.norm(rp, axis=1, keepdims=True)
        b += MU0_4PI * src_vol[i] * np.cross(src_jdensity[i, :], rp) / rmag**3

    return b


# Testing the functions first expected
# value using all elements as source is ~0.45 T in vertical direction
targets = np.array([[0.0, 0.0, 0.0]])
# print(bfield(mesh.centroids, mesh.volumes, jdensity, targets))


def bfield_monopole(src_pts, src_vol, src_jdensity, targets):
    src_moments = src_jdensity * src_vol[:, np.newaxis]
    src_moments_mag = np.linalg.norm(src_moments, axis=1)
    centroid = np.sum(
        (src_pts * src_moments_mag[:, np.newaxis]) / np.sum(src_moments_mag), axis=0
    )

    jv = np.sum(src_moments, axis=0)
    rp = targets - centroid
    rmag = np.linalg.norm(rp, axis=1, keepdims=True)
    return MU0_4PI * np.cross(jv, rp) / rmag**3


def dipole_expansion(src_pts, src_vol, src_jdensity, center):
    # returns m and a 3x3 tensor M
    M = np.zeros((3, 3))

    # Doing this as a loop because the Rust version will also be in a loop
    n = src_pts.shape[0]
    for e in range(n):
        for i in range(3):
            for j in range(3):
                M[i, j] += src_vol[e] * src_jdensity[e, i] * (src_pts[e, j] - center[j])

    m = 0.5 * np.array(
        [
            M[2, 1] - M[1, 2],
            M[0, 2] - M[2, 0],
            M[1, 0] - M[0, 1],
        ]
    )
    return m, M


def bfield_dipole(
    src_pts,
    src_vol,
    src_jdensity,
    targets,
    expansion: Literal["full", "antisymmetric"] = "full",
):
    # monopole + dipole correction

    src_moments = src_jdensity * src_vol[:, np.newaxis]
    src_moments_mag = np.linalg.norm(src_moments, axis=1)
    centroid = np.sum(
        (src_pts * src_moments_mag[:, np.newaxis]) / np.sum(src_moments_mag), axis=0
    )

    src_rp = src_pts - centroid

    # Total dipole moment of the tree node
    m, M = dipole_expansion(src_pts, src_vol, src_jdensity, centroid)

    b_monopole = bfield_monopole(src_pts, src_vol, src_jdensity, targets)
    rp = targets - centroid
    rmag = np.linalg.norm(rp, axis=1, keepdims=True)
    rhat = rp / np.linalg.norm(rp, axis=1, keepdims=True)

    if expansion == "full":
        return b_monopole + MU0_4PI * (2.0*m - 3.0*np.cross(rhat, rhat @ M.T)) / rmag**3

    else:
        return b_monopole + MU0_4PI * (3.0 * rhat * (rhat @ m)[:,None] - m) / rmag**3


# bfield_monopole(mesh.centroids, mesh.volumes, jdensity, targets)
n = 1000
targets = np.zeros((n, 3))
targets[:, 1] = -0.128
targets[:, 2] = -0.1

# Box size is approximately 0.2 side length with centroid 0.1
theta = np.linspace(0.1, 1.0, n)
targets[:, 0] = 0.2 / theta - 0.1


bf = bfield_all(src_pts, src_vol, src_jdensity, targets)
bm = bfield_monopole(src_pts, src_vol, src_jdensity, targets)
bda = bfield_dipole(src_pts, src_vol, src_jdensity, targets, expansion="antisymmetric")
bdf = bfield_dipole(src_pts, src_vol, src_jdensity, targets, expansion="full")

err_bm = np.abs(bm - bf)
err_bda = np.abs(bda - bf)
err_bdf = np.abs(bdf - bf)

# bf_mag = np.linalg.norm(bf, axis=1)
bm_mag = np.linalg.norm(err_bm, axis=1)
bda_mag = np.linalg.norm(err_bda, axis=1)
bdf_mag = np.linalg.norm(err_bdf, axis=1)

import matplotlib.pyplot as plt 

fig, ax = plt.subplots() 
# ax.plot(theta, bf_mag,label="Full O(N^2)")
ax.plot(theta, bm_mag, label="Monopole")
ax.plot(theta,bda_mag,label="Dipole, Antisymmetric")
ax.plot(theta,bdf_mag,label="Dipole, Full")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()
ax.set_xlabel("Theta")
ax.set_ylabel("Error Magnitude, |B| (T)")
fig.savefig("tests/fig/test_multipole.png")
