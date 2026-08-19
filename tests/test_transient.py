import oersted
# import numpy as np

major_radius = 0.10
minor_radius = 0.05
wall_thickness = 0.01
mesh_size = 0.015
torus = oersted.testing.make_torus(
    major_radius, minor_radius, wall_thickness, mesh_size
)
print(f"Mesh size: {torus.num_elems} elements")


ring_current = 1e6
b = 0.01
h = 0.01
ring_jmag = ring_current / (b * h)
ring, ring_jdensity = oersted.testing.make_ring(
    major_radius, 0.0, b, h, b, jmag=ring_jmag
)

mesh = torus.append(ring)

oersted.mesh.plot_mesh(mesh, transparency=True)
