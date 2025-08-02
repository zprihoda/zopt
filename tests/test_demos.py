import os
import pytest
import runpy

from glob import glob

testDir = os.path.dirname(__file__)
demoPattern = os.path.join(testDir, "..", "demos", "[!_]*.py")
demoList = [os.path.basename(demo).replace(".py","") for demo in glob(demoPattern)]

@pytest.mark.slow
@pytest.mark.parametrize("demo", demoList)
def test_demos(monkeypatch, demo):
    try:
        monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
        runpy.run_module("demos." + demo, run_name="__main__")
    except Exception as e:
        pytest.fail(f"'{demo}' failed during execution: {e}")
