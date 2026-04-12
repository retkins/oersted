"""Run examples to make sure they don't go stale"""

import os
import pathlib

# import runpy
import subprocess
import sys
import pytest


EXAMPLES_DIR = pathlib.Path(__file__).parent / "../examples"
EXAMPLES = sorted(
    [
        EXAMPLES_DIR / x
        for x in os.listdir(EXAMPLES_DIR)
        if (os.path.isfile(EXAMPLES_DIR / x) and x.endswith(".py"))
    ]
)


@pytest.mark.parametrize("fp", EXAMPLES)
def test_example(fp: pathlib.Path):
    try:
        print(f"Running example: {fp}", flush=True)
        result = subprocess.run(
            [sys.executable, str(fp)], capture_output=True, text=True
        )

        # mod = runpy.run_path(str(fp), run_name="__test__")
        # if "main" in mod:
        #     mod["main"]()
        assert result.returncode == 0, result.stderr
    except:
        print(f"Failed to run example {fp}")
        raise


if __name__ == "__main__":
    for ex in EXAMPLES:
        test_example(ex)
