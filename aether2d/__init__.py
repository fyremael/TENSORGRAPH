"""AETHER2D compatibility module aliasing tensorgraph."""

import sys
import importlib
import pkgutil
import tensorgraph

sys.modules["aether2d"] = tensorgraph

for _, module_name, is_pkg in pkgutil.walk_packages(tensorgraph.__path__, tensorgraph.__name__ + "."):
    try:
        mod = importlib.import_module(module_name)
        alias_name = module_name.replace("tensorgraph", "aether2d", 1)
        sys.modules[alias_name] = mod
    except Exception:
        pass

# Re-export tensorgraph attributes
for attr in dir(tensorgraph):
    if not attr.startswith("__"):
        globals()[attr] = getattr(tensorgraph, attr)
