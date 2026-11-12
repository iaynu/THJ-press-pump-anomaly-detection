import importlib
import pkgutil

import pytest

import kamp
import kamp.experiments

MODULES = [m.name for m in pkgutil.iter_modules(kamp.__path__, "kamp.") if not m.ispkg] + \
          [m.name for m in pkgutil.iter_modules(kamp.experiments.__path__, "kamp.experiments.")]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_inference_core_does_not_depend_on_experiments():
    """analysis / dashboard_core / models must stay importable without the experiment code"""
    import sys
    for n in [m for m in sys.modules if m.startswith("kamp.experiments")]:
        del sys.modules[n]
    importlib.reload(importlib.import_module("kamp.analysis"))
    importlib.reload(importlib.import_module("kamp.dashboard_core"))
    assert not [m for m in sys.modules if m.startswith("kamp.experiments")]
