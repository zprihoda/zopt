import os
import pytest
import runpy

from glob import glob

testDir = os.path.dirname(__file__)
demoPattern = os.path.join(testDir, "..", "demos", "[!_]*.py")
demoList = glob(demoPattern)


@pytest.mark.slow
@pytest.mark.parametrize("demo", demoList)
def test_demos(monkeypatch, demo):
    try:
        # Import the script as a module to execute its top-level code
        demo_name = os.path.basename(demo).replace(".py", "")
        monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
        runpy.run_module("demos." + demo_name, run_name="__main__")
    except Exception as e:
        pytest.fail(f"'{demo_name}' failed during execution: {e}")
