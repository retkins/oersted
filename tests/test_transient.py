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
torus.plot()
