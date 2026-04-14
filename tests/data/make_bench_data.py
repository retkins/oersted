"""Make data for benches"""

import numpy as np 
import oersted 

mesh = oersted.Mesh.from_step("tests/data/loop_magnet.stp", 40e-3)

np.savetxt("tests/data/bench.nodes", mesh.nodes, delimiter=',')
np.savetxt("tests/data/bench.connectivity", mesh.connectivity, delimiter=',')
