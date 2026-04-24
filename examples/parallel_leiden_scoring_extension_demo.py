#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import networkit as nk


def build_demo_graph():
    graph = nk.Graph(6, weighted=False, directed=False)
    for u, v in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]:
        graph.addEdge(u, v)
    return graph


def parse_args():
    parser = argparse.ArgumentParser(
        description="Demo ParallelLeidenView's shared-library move scoring extension mechanism."
    )
    parser.add_argument(
        "--plugin",
        type=Path,
        help=(
            "Path to a scorer shared library. "
            "Example: build/networkit/cpp/community/libnetworkit_parallel_leiden_modularity_extension.dylib"
        ),
    )
    parser.add_argument(
        "--use-env",
        action="store_true",
        help="Load the plugin through NETWORKIT_LEIDEN_MOVE_SCORING_LIB instead of the Python API.",
    )
    parser.add_argument("--iterations", type=int, default=2, help="Number of Leiden iterations.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Resolution parameter.")
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Randomize node order. Disabled by default to keep output stable.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    graph = build_demo_graph()
    plugin_loaded_via = None

    if args.use_env:
        if args.plugin is None:
            raise SystemExit("--use-env requires --plugin")
        os.environ["NETWORKIT_LEIDEN_MOVE_SCORING_LIB"] = str(args.plugin.resolve())
        plugin_loaded_via = "env"

    leiden = nk.community.ParallelLeidenView(
        graph, iterations=args.iterations, randomize=args.randomize, gamma=args.gamma
    )

    if args.plugin is not None and not args.use_env:
        if hasattr(leiden, "loadMoveScoringExtension"):
            leiden.loadMoveScoringExtension(str(args.plugin.resolve()))
            plugin_loaded_via = "api"
        else:
            # Fallback for older Python extension builds where the C++ env hook exists
            # but the wrapper method has not been rebuilt yet.
            os.environ["NETWORKIT_LEIDEN_MOVE_SCORING_LIB"] = str(args.plugin.resolve())
            leiden = nk.community.ParallelLeidenView(
                graph, iterations=args.iterations, randomize=args.randomize, gamma=args.gamma
            )
            plugin_loaded_via = "env-fallback"

    leiden.run()
    partition = leiden.getPartition()

    print("communities:", partition.numberOfSubsets())
    print("assignment:", [partition[i] for i in range(graph.numberOfNodes())])
    if args.plugin is None:
        print("scorer: built-in modularity")
    elif plugin_loaded_via == "api":
        print("scorer: plugin loaded via ParallelLeidenView.loadMoveScoringExtension()")
    elif plugin_loaded_via == "env":
        print("scorer: plugin loaded from NETWORKIT_LEIDEN_MOVE_SCORING_LIB")
    elif plugin_loaded_via == "env-fallback":
        print("scorer: plugin loaded from NETWORKIT_LEIDEN_MOVE_SCORING_LIB")
        print("note: Python wrapper method not available in this build, used env fallback")
    else:
        print("scorer: plugin requested, but no loading path was used")


if __name__ == "__main__":
    main()
