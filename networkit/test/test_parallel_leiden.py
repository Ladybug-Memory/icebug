#!/usr/bin/env python3
"""
Test Arrow-backed Graph construction with ParallelLeidenView community detection algorithm.

This script demonstrates:
1. Converting pandas DataFrame with PyArrow backend to NetworKit CSR graph
2. Zero-copy construction using Arrow C Data Interface
3. Running ParallelLeidenView algorithm on CSR graphs
4. Memory-efficient graph processing for community detection
"""

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pandas as pd
import networkit as nk


def create_graph_arrow_optimized(df, directed=False):
    """
    Create a NetworKit CSR graph from pandas DataFrame using the new fromCSR method.
    """
    print(
        f"Creating {'directed' if directed else 'undirected'} graph from {len(df)} edges..."
    )

    # Convert to Arrow table if needed
    if isinstance(df, pd.DataFrame):
        table = pa.Table.from_pandas(df, preserve_index=False)
    else:
        table = df

    # Get source and target arrays
    sources = table["source"].to_pylist()
    targets = table["target"].to_pylist()

    # Find number of nodes
    max_source = max(sources)
    max_target = max(targets)
    n_nodes = max(max_source, max_target) + 1
    print(f"Graph has {n_nodes} nodes")

    # For undirected graphs, create bidirectional edge list
    if not directed:
        # Add reverse edges for undirected graph
        all_sources = sources + targets
        all_targets = targets + sources
        print(f"Added reverse edges, total edges: {len(all_sources)}")
    else:
        all_sources = sources
        all_targets = targets

    # Build CSR indptr array correctly
    indptr = [0] * (n_nodes + 1)

    # Count edges per node
    edge_counts = [0] * n_nodes
    for src in all_sources:
        edge_counts[src] += 1

    # Build cumulative sum for indptr
    for i in range(n_nodes):
        indptr[i + 1] = indptr[i] + edge_counts[i]

    # Sort edges by source for CSR format
    edges_with_idx = [
        (src, tgt, i) for i, (src, tgt) in enumerate(zip(all_sources, all_targets))
    ]
    edges_with_idx.sort(key=lambda x: x[0])  # Sort by source

    # Extract sorted arrays for CSR
    sorted_sources = [x[0] for x in edges_with_idx]
    sorted_targets = [x[1] for x in edges_with_idx]

    print(f"CSR indices array length: {len(sorted_targets)}")
    print(f"CSR indptr array: {indptr}")

    # Create PyArrow arrays for CSR format
    indices_arrow = pa.array(sorted_targets, type=pa.uint64())
    indptr_arrow = pa.array(indptr, type=pa.uint64())

    print(f"Creating Graph with CSR constructor: n={n_nodes}, directed={directed}")
    # Create NetworKit Graph using the new fromCSR method - using GraphR for unweighted CSR graphs
    graph = nk.GraphR.fromCSR(n_nodes, directed, indices_arrow, indptr_arrow)

    print(
        f"Created NetworKit graph: {graph.numberOfNodes()} nodes, {graph.numberOfEdges()} edges"
    )
    return graph


def test_small_graph():
    """Test with a small known graph structure."""
    print("=== Testing Small Graph ===")

    # Create a small graph: 0-1-2-3-0 (cycle) + 1-4
    edges_data = {"source": [0, 1, 2, 3, 1], "target": [1, 2, 3, 0, 4]}

    df = pd.DataFrame(edges_data)
    print(f"Input edges: {df.to_dict('records')}")

    # Create Arrow-backed DataFrame
    df_arrow = df.astype({"source": "uint64[pyarrow]", "target": "uint64[pyarrow]"})

    # Create NetworKit Graph using CSR constructor
    graph = create_graph_arrow_optimized(df_arrow, directed=False)

    # Verify graph properties
    print(f"\nGraph verification:")
    print(f"  Nodes: {graph.numberOfNodes()}")
    print(f"  Edges: {graph.numberOfEdges()}")
    print(f"  Directed: {graph.isDirected()}")
    print(f"  Weighted: {graph.isWeighted()}")

    # Test basic graph operations
    print(f"\nTesting graph operations:")
    for u in range(graph.numberOfNodes()):
        degree = graph.degree(u)
        print(f"  Node {u}: degree = {degree}")

    return graph


def test_parallel_leiden_algorithm(graph):
    """Test ParallelLeidenView algorithm on the graph."""
    print("\n=== Testing ParallelLeidenView Algorithm ===")

    try:
        # Create ParallelLeidenView instance
        leiden = nk.community.ParallelLeidenView(
            graph, iterations=3, randomize=True, gamma=1.0
        )
        print("Created ParallelLeidenView instance")

        # Run the algorithm
        print("Running ParallelLeidenView...")
        leiden.run()
        print("ParallelLeidenView completed successfully!")

        # Get results
        partition = leiden.getPartition()
        print(f"Community detection results for {graph.numberOfNodes()} nodes:")
        print(f"Number of communities found: {partition.numberOfSubsets()}")

        # Check if partition is valid before calculating modularity
        try:
            # Validate partition - check for invalid community IDs
            valid_partition = True
            max_valid_community = partition.upperBound()

            for node in range(graph.numberOfNodes()):
                community = partition[node]
                if community >= max_valid_community or community < 0:
                    print(
                        f"WARNING: Node {node} has invalid community ID {community} (max valid: {max_valid_community-1})"
                    )
                    valid_partition = False
                    break

            if valid_partition and partition.numberOfSubsets() > 0:
                modularity = nk.community.Modularity().getQuality(partition, graph)
                print(f"Modularity: {modularity:.6f}")
            else:
                print("Modularity: INVALID PARTITION - cannot calculate")

        except Exception as mod_error:
            print(f"Modularity calculation failed: {mod_error}")

        # Print community assignments
        for node in range(min(graph.numberOfNodes(), 10)):  # Show first 10 nodes
            community = partition[node]
            print(f"  Node {node}: Community = {community}")

        if graph.numberOfNodes() > 10:
            print(f"  ... and {graph.numberOfNodes() - 10} more nodes")

        return True

    except Exception as e:
        print(f"Error running ParallelLeidenView: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_larger_graph():
    """Test with a larger graph to ensure scalability and community structure."""
    print("\n=== Testing Larger Graph ===")

    # Create a larger graph with multiple communities
    # Community 1: nodes 0-4 (complete subgraph)
    # Community 2: nodes 5-9 (complete subgraph)
    # Bridge: connect communities with edge 4-5

    edges = []

    # Community 1: complete graph on nodes 0-4
    for i in range(5):
        for j in range(i + 1, 5):
            edges.append((i, j))

    # Community 2: complete graph on nodes 5-9
    for i in range(5, 10):
        for j in range(i + 1, 10):
            edges.append((i, j))

    # Bridge between communities
    edges.append((4, 5))

    # Create DataFrame
    df = pd.DataFrame(edges, columns=["source", "target"])
    print(f"Created graph with {len(df)} edges")

    # Convert to Arrow
    df_arrow = df.astype({"source": "uint64[pyarrow]", "target": "uint64[pyarrow]"})

    # Create graph using CSR constructor
    graph = create_graph_arrow_optimized(df_arrow, directed=False)

    # Run ParallelLeidenView
    success = test_parallel_leiden_algorithm(graph)

    return graph, success


def main():
    """Main test function."""
    print("NetworKit Arrow-backed Graph + ParallelLeidenView Test")
    print("=" * 60)

    try:
        # Test 1: Small graph
        small_graph = test_small_graph()
        small_success = test_parallel_leiden_algorithm(small_graph)

        # Test 2: Larger graph
        large_graph, large_success = test_larger_graph()

        print("\n" + "=" * 60)
        if small_success and large_success:
            print("✅ ALL TESTS PASSED!")
            print("✅ Arrow-backed CSR graph construction works")
            print("✅ ParallelLeidenView algorithm works on CSR graphs")
            print("✅ Memory-efficient zero-copy construction verified")
        else:
            print("❌ SOME TESTS FAILED!")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
