"""Compatibility alias for the :mod:`networkit` package.

This allows users to write ``import icebug as ib`` and use the same API that is
available via ``import networkit as nk``.
"""

import importlib
import sys

import networkit as _networkit

# Expose the exact same top-level module object under the "icebug" name.
sys.modules[__name__] = _networkit

# Keep submodule aliasing explicit and static.
_ALIASED_SUBMODULES = [
    "graph",
    "graphio",
    "community",
    "centrality",
    "generators",
    "structures",
    "engineering",
    "embedding",
    "distance",
    "components",
    "gephi",
    "coloring",
    "flow",
    "sparsification",
    "scd",
    "clique",
    "globals",
    "linkprediction",
    "correlation",
    "matching",
    "coarsening",
    "reachability",
    "simulation",
    "stats",
    "viz",
    "randomization",
    "independentset",
    "graphtools",
    "support",
    "plot",
    "profiling",
    "nxadapter",
]

for _submodule in _ALIASED_SUBMODULES:
    try:
        _module = importlib.import_module(f"networkit.{_submodule}")
    except ImportError:
        # Optional dependencies may make some modules unavailable.
        continue
    sys.modules[f"icebug.{_submodule}"] = _module
