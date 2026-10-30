#!/usr/bin/env python3

from pathlib import Path

import numpy as np

import networkit as nk
from networkit.scd import ShellStruct


def build_demo_graph():
    # a chain of 3-clique, 4-clique, 3-clique, 4-clique
    src = [0, 0, 1, 2, 3, 3, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 10, 11, 11, 12]
    tgt = [1, 2, 2, 3, 4, 5, 6, 5, 6, 6, 7, 8, 9, 9, 10, 11, 12, 13, 12, 13, 13]
    G = nk.Graph(14, directed=False)
    for s, t in zip(src, tgt):
        G.addEdge(s, t)
    coredecomp = np.array([2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 3, 3, 3, 3], dtype=np.uint64)
    return G, coredecomp


def main():
    components = Path("output") / "shellstruct.components.feather"
    tree = Path("output") / "shellstruct.tree.feather"

    graph, coredecomp = build_demo_graph()
    shell = ShellStruct(graph)

    if components.exists() and tree.exists():
        print("Loading shellstruct...")
        shell.load(components, tree)
    else:
        print("Building shellstruct...")
        shell.build(coredecomp)

    seed = {0, 1}
    comm = shell.expandOneCommunity(seed)
    members = shell.community(seed)
    print(f"Comm for {seed}:", comm)  # 0, 1 connected 2-shell but not 3-shell
    print(f"Comm for {seed} has size {len(members)} and coreness {shell.score(seed)}")

    seed = {3, 4}
    comm = shell.expandOneCommunity(seed)
    members = shell.community(seed)
    print(f"Comm for {seed}:", comm)  # 3 is in 3-shell, returns just the 4-clique
    print(f"Comm for {seed} has size {len(members)} and coreness {shell.score(seed)}")

    seed = {3, 13}
    comm = shell.expandOneCommunity(seed)
    members = shell.community(seed)
    print(f"Comm for {seed}:", comm)  # 3, 13 are connected in 2-shell (but not 3-shell)
    print(f"Comm for {seed} has size {len(members)} and coreness {shell.score(seed)}")

    if not components.exists() or not tree.exists():
        components.parent.mkdir(exist_ok=True)
        shell.save(components, tree, compression="ZSTD")


if __name__ == "__main__":
    main()
