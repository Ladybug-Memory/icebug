#!/usr/bin/env python3

"""Compare the preserved ParK implementation with the SIGMOD 2025 implementation."""

import argparse
import math
import statistics
import time

import networkit as nk


def validate(algorithm, expected_scores, expected_max_core, name):
    actual_scores = algorithm.scores()
    actual_max_core = algorithm.maxCoreNumber()
    if actual_max_core != expected_max_core:
        raise RuntimeError(
            f"{name} returned max core {actual_max_core}; expected {expected_max_core}")
    if len(actual_scores) != len(expected_scores):
        raise RuntimeError(
            f"{name} returned {len(actual_scores)} scores; expected {len(expected_scores)}")
    if actual_scores != expected_scores:
        mismatch = next(
            node for node, (expected, actual) in enumerate(zip(expected_scores, actual_scores))
            if expected != actual)
        raise RuntimeError(
            f"{name} returned core {actual_scores[mismatch]} for node {mismatch}; "
            f"expected {expected_scores[mismatch]}")


def measure(name, factory, expected_scores, expected_max_core, repeats):
    samples = []
    for _ in range(repeats):
        algorithm = factory()
        started = time.perf_counter()
        algorithm.run()
        samples.append(time.perf_counter() - started)
        validate(algorithm, expected_scores, expected_max_core, name)
    return statistics.median(samples), samples


def benchmark(name, graph, repeats):
    oracle = nk.centrality.CoreDecomposition(
        graph, normalized=False, enforceBucketQueueAlgorithm=True)
    oracle.run()
    expected_scores = oracle.scores()
    expected_max_core = oracle.maxCoreNumber()

    park_median, park_samples = measure(
        "ParK", lambda: nk.centrality.CoreDecomposition(graph), expected_scores,
        expected_max_core, repeats)
    parallel_median, parallel_samples = measure(
        "SIGMOD25", lambda: nk.centrality.ParallelCoreDecomposition(graph), expected_scores,
        expected_max_core, repeats)

    print(f"\n{name}: n={graph.numberOfNodes()} m={graph.numberOfEdges()}")
    print(
        f"validated={2 * repeats} timed runs against exact per-node scores; "
        f"max_core={expected_max_core}")
    print(f"ParK: median={park_median:.6f}s samples={park_samples}")
    print(f"SIGMOD25: median={parallel_median:.6f}s samples={parallel_samples}")
    print(f"ParK/SIGMOD25={park_median / parallel_median:.3f}x")


def sparse_er():
    nk.setSeed(42, False)
    return nk.generators.ErdosRenyiGenerator(200_000, 0.00005, False).generate()


def vgc_path():
    graph = nk.Graph(1_000_000)
    for node in range(1, graph.numberOfNodes()):
        graph.addEdge(node - 1, node)
    return graph


def sampling_star():
    graph = nk.Graph(1_000_001)
    for leaf in range(1, graph.numberOfNodes()):
        graph.addEdge(0, leaf)
    return graph


def sampling_bipartite():
    graph = nk.Graph(20_020)
    for high_degree_node in range(20):
        for low_degree_node in range(20, graph.numberOfNodes()):
            graph.addEdge(high_degree_node, low_degree_node)
    return graph


def hbs_clique():
    return nk.generators.ErdosRenyiGenerator(2_000, 1.0, False).generate()


def graph_from_symmetric_csr(indices, indptr):
    """Build an undirected zero-copy GraphR from NumPy uint64 CSR storage."""
    import pyarrow as pa

    return nk.Graph.fromCSR(
        len(indptr) - 1,
        False,
        pa.array(indices, type=pa.uint64()),
        pa.array(indptr, type=pa.uint64()),
    )


def large_bipartite(target_edges):
    """Build K(left, right) with exactly ``target_edges`` when divisible by left."""
    import numpy as np

    left = 1_000
    if target_edges < left or target_edges % left:
        raise ValueError(f"large bipartite edge count must be divisible by {left}")
    right = target_edges // left
    nodes = left + right
    adjacency_entries = 2 * target_edges

    indices = np.empty(adjacency_entries, dtype=np.uint64)
    right_nodes = np.arange(left, nodes, dtype=np.uint64)
    for node in range(left):
        begin = node * right
        indices[begin:begin + right] = right_nodes

    left_nodes = np.arange(left, dtype=np.uint64)
    reverse_begin = target_edges
    rows_per_chunk = 16_384
    for row_begin in range(0, right, rows_per_chunk):
        row_end = min(row_begin + rows_per_chunk, right)
        begin = reverse_begin + row_begin * left
        end = reverse_begin + row_end * left
        indices[begin:end].reshape(row_end - row_begin, left)[:] = left_nodes

    indptr = np.empty(nodes + 1, dtype=np.uint64)
    indptr[:left + 1] = np.arange(left + 1, dtype=np.uint64) * right
    indptr[left + 1:] = target_edges + np.arange(1, right + 1, dtype=np.uint64) * left
    return graph_from_symmetric_csr(indices, indptr)


def large_clique(target_edges):
    """Build the largest complete graph with no more than ``target_edges`` edges."""
    import numpy as np

    nodes = (1 + math.isqrt(1 + 8 * target_edges)) // 2
    edges = nodes * (nodes - 1) // 2
    adjacency_entries = 2 * edges
    indices = np.empty(adjacency_entries, dtype=np.uint64)
    all_nodes = np.arange(nodes, dtype=np.uint64)
    row_width = nodes - 1
    for node in range(nodes):
        begin = node * row_width
        indices[begin:begin + node] = all_nodes[:node]
        indices[begin + node:begin + row_width] = all_nodes[node + 1:]

    indptr = np.arange(nodes + 1, dtype=np.uint64) * row_width
    return graph_from_symmetric_csr(indices, indptr)


def large_sparse_er(target_edges):
    """Build a sparse Erdos-Renyi graph with about ``target_edges`` edges."""
    nodes = max(1_000, target_edges // 20)
    probability = (2.0 * target_edges) / (nodes * (nodes - 1))
    if probability > 1.0:
        raise ValueError("large sparse ER target is too dense for the selected node count")
    nk.setSeed(42, False)
    return nk.generators.ErdosRenyiGenerator(nodes, probability, False).generate()


def large_star(target_edges):
    """Build a zero-copy star with exactly ``target_edges`` edges."""
    import numpy as np

    nodes = target_edges + 1
    indices = np.empty(2 * target_edges, dtype=np.uint64)
    indices[:target_edges] = np.arange(1, nodes, dtype=np.uint64)
    indices[target_edges:] = 0

    indptr = np.empty(nodes + 1, dtype=np.uint64)
    indptr[0] = 0
    indptr[1:] = target_edges + np.arange(nodes, dtype=np.uint64)
    return graph_from_symmetric_csr(indices, indptr)


def large_path(target_edges):
    """Build a zero-copy path with exactly ``target_edges`` edges."""
    import numpy as np

    nodes = target_edges + 1
    indices = np.empty(2 * target_edges, dtype=np.uint64)
    indices[0] = 1
    if nodes > 2:
        internal_nodes = np.arange(1, nodes - 1, dtype=np.uint64)
        indices[1:-1:2] = internal_nodes - 1
        indices[2:-1:2] = internal_nodes + 1
    indices[-1] = nodes - 2

    indptr = np.empty(nodes + 1, dtype=np.uint64)
    indptr[0] = 0
    indptr[1:nodes] = 2 * np.arange(1, nodes, dtype=np.uint64) - 1
    indptr[nodes] = 2 * target_edges
    return graph_from_symmetric_csr(indices, indptr)


CASES = {
    "sparse-er": sparse_er,
    "vgc-path": vgc_path,
    "sampling-star": sampling_star,
    "sampling-bipartite": sampling_bipartite,
    "hbs-clique": hbs_clique,
}

LARGE_CASES = {
    "large-bipartite": large_bipartite,
    "large-clique": large_clique,
    "large-sparse-er": large_sparse_er,
    "large-star": large_star,
    "large-path": large_path,
}

LARGE_ALL_CASES = ("large-bipartite", "large-clique")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=["all", "large-all", *CASES, *LARGE_CASES], default="all")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=15)
    parser.add_argument(
        "--large-edges", type=int, default=100_000_000,
        help="target undirected edge count for large CSR cases (default: 100 million)")
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.large_edges < 1:
        parser.error("--large-edges must be positive")

    nk.setNumberOfThreads(args.threads)
    if args.case == "all":
        selected = [(name, factory) for name, factory in CASES.items()]
    elif args.case == "large-all":
        selected = [(name, LARGE_CASES[name]) for name in LARGE_ALL_CASES]
    elif args.case in LARGE_CASES:
        selected = [(args.case, LARGE_CASES[args.case])]
    else:
        selected = [(args.case, CASES[args.case])]
    for name, factory in selected:
        graph = factory(args.large_edges) if name in LARGE_CASES else factory()
        benchmark(name, graph, args.repeats)


if __name__ == "__main__":
    main()
