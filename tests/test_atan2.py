import oersted
import numpy as np

n = 1_000_000
yrange = (-1e6, 1e6)
xrange = (-1e6, 1e6)


def test_atan2(verbose=False):

    y = np.random.uniform(low=yrange[0], high=yrange[1], size=n)
    x = np.random.uniform(low=xrange[0], high=xrange[1], size=n)

    # Oersted
    result_oersted = oersted.atan2(y, x)

    # numpy
    result_numpy = np.atan2(y, x)

    err = result_oersted - result_numpy
    abs_err = np.abs(err)
    max_err = np.max(abs_err)
    min_err = np.min(abs_err)
    mean_err = np.mean(abs_err)

    if verbose:
        print(f"max err: {max_err}")
        print(f"min err: {min_err}")
        print(f"avg err: {mean_err}")

    assert max_err < 2e-6


if __name__ == "__main__":
    test_atan2(verbose=True)
