import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "adapter,blocked",
    [
        (None, ["pandas", "polars", "numpy"]),
        ("pandas", ["polars"]),
        ("polars", ["pandas", "numpy"]),
    ],
)
def test_optional_dependencies_do_not_leak(adapter, blocked):
    code = f"""
import importlib.abc
import sys

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {blocked!r}:
            raise ModuleNotFoundError(fullname, name=fullname)

sys.meta_path.insert(0, Block())
import jevframe
assert not any(name in sys.modules for name in {blocked!r})
adapter = {adapter!r}
if adapter:
    __import__('jevframe.' + adapter)
assert not any(name in sys.modules for name in {blocked!r})
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_root_import_does_not_register_accessors():
    code = """
import pandas as pd
import polars as pl
import jevframe
assert not hasattr(pd.DataFrame(), 'jev')
assert not hasattr(pl.DataFrame(), 'jev')
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
