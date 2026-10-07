import oersted

from pathlib import Path

here = Path(__file__).parent

f = here / "data" / "torus-45deg.step"

mesh = oersted.Mesh.from_step(f, 5e-3, n_sectors=8)
mesh.plot()
