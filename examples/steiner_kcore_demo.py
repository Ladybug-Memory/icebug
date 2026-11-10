#!/usr/bin/env python3

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import networkit as nk
from networkit.scd import SteinerKCore


def build_demo_graph():
    graph = nk.Graph(11, weighted=False, directed=False)
    for u, v in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
        graph.addEdge(u, v)
    for u, v in [(4, 5), (4, 6), (5, 6)]:
        graph.addEdge(u, v)
    for u, v in [(7, 8), (7, 9), (7, 10), (8, 9), (8, 10), (9, 10)]:
        graph.addEdge(u, v)
    graph.addEdge(3, 4)
    graph.addEdge(6, 7)

    coreness = [3, 3, 3, 3, 2, 2, 2, 3, 3, 3, 3]
    return graph, coreness


def run_parallel(steiner, queries, num_workers, threads_per_worker, max_batch_size=None):
    """Run many queries across threads, one contiguous batch per worker.

    Groups the queries into <= num_workers batches (optional max_batch_size), and executes
    steiner.communities with threads_per_worker threads. Results are returned in input order.
    """
    n = len(queries)
    if n == 0:
        return [], np.empty(0, dtype=np.uint64)
    num_workers = max(1, min(num_workers, n))

    batch_size = math.ceil(n / num_workers)
    if max_batch_size is not None:
        batch_size = min(batch_size, max_batch_size)
    batches = [queries[i : i + batch_size] for i in range(0, n, batch_size)]

    # the OpenMP thread count is per calling thread, so each worker sets its own
    with ThreadPoolExecutor(
        num_workers, initializer=nk.setNumberOfThreads, initargs=(threads_per_worker,)
    ) as pool:
        results = list(pool.map(steiner.communities, batches))
    members = [comm for batch_members, _ in results for comm in batch_members]
    coreness = np.concatenate([batch_coreness for _, batch_coreness in results])
    return members, coreness


def show(queries, members, coreness):
    for seed, comm, k in zip(queries, members, coreness):
        print(f"  {seed=}: k={k}, community={sorted(comm.tolist())}")


def main():
    graph, coreness = build_demo_graph()
    steiner = SteinerKCore(graph, np.array(coreness, dtype=np.uint64))

    seed = {0, 1, 2}
    print(f"{seed=}:", steiner.expandOneCommunity(seed))
    seed = {4}
    print(f"{seed=}:", steiner.expandOneCommunity(seed))
    seed = {2, 9}
    print(f"{seed=}:", steiner.expandOneCommunity(seed))

    queries = [{0, 1, 2}, {4}, {2, 9}]
    print("Running all three as a batch:")
    show(queries, *steiner.communities(queries))
    print("Running as two parallel batches:")
    show(queries, *run_parallel(steiner, queries, 2, 2))


if __name__ == "__main__":
    main()
